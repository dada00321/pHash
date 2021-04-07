import numpy as np
from io import BytesIO
from PIL import Image
import cv2 

class PHash:
    def __init__(self, n_grid_points=9, 
                 crop_percentiles=(5,95), 
                 is_larger_crop=False,
                 P=None, diagonal_neighbors=True,
                 identical_tolerance=2/255,
                 n_levels=2):
        
        if crop_percentiles is None:
            crop_percentiles = (0, 100)
        
        assert all((type(crop_percentiles) is tuple,
                    type(crop_percentiles[0]) is int,
                    type(crop_percentiles[1]) is int,
                    type(is_larger_crop) is bool,
                    type(n_grid_points) is int,
                    type(P) is int or P is None,
                    type(diagonal_neighbors) is bool,
                    type(identical_tolerance) in (int, float),
                    type(n_levels) is int)),\
               "type of `crop_percentiles` should be `tuple`; "+\
               "type of elements in `crop_percentiles` should be `int`; "+\
               "type of `is_larger_crop` should be `bool`; "+\
               "type of `n_grid_points` should be `int`; "+\
               "type of `diagonal_neighbors` should be `bool`; "+\
               "type of `identical_tolerance` should be `int`; "
        
        if all((P is not None, crop_percentiles is not None,
               identical_tolerance is not None)):
            assert all((crop_percentiles[0] < crop_percentiles[1],
                        crop_percentiles[0] >= 0,
                        crop_percentiles[1] <= 100,
                        P >= 1,
                        identical_tolerance in range(0, 2),
                        n_levels > 0)),\
                   "values of crop_percentiles[0], crop_percentiles[1] "+\
                   "should in range: [0, 100]; "+\
                   "value of `n_grid_points` should larger than or equal to 0; "+\
                   "value of `P` should larger than or equal to 1; "+\
                   "value of `identical_tolerance` should in range: [0, 1]; "

        self.crop_percentiles = crop_percentiles
        self.lower_bound = crop_percentiles[0]
        self.upper_bound = crop_percentiles[1]
        self.is_larger_crop = is_larger_crop
        self.n_grid_points = n_grid_points
        self.P = P
        self.diagonal_neighbors = diagonal_neighbors
        self.identical_tolerance = identical_tolerance
        self.n_levels = n_levels
        
    def _get_boundaries_of_cropped_image(self, img, lower_bound=5, upper_bound=95, is_larger_crop=False):
        """
        rtn_val: boundaries (<list of 2 tuples>)
            (2 is for row lower & upper boundaries,
             2 is for column lower & upper boundaries)
            (default) [(r1, r2), (c1, c2)] 
                        --> "r1 <= r2" and "c1 <= c2"
            e.g,      [(36, 684), (24, 452)]
        """
        ##########
        # cusum_of_each_col: Cumulative sum of each col
        #   --> obtained by calculating differences of nearby rows pairwise
        # cusum_of_each_row: Cumulative sum of each row
        #   --> obtained by calculating differences of nearby columns pairwise
        ##########
        cusum_of_each_col = np.cumsum(np.sum(np.abs(np.diff(img, axis=0)), axis=0))
        cusum_of_each_row = np.cumsum(np.sum(np.abs(np.diff(img, axis=1)), axis=1))
        # compute percentiles
        '''
        c1: 
            Get the "left" boundary,
            if and only if
            get the lower percentile of 
            `cusum_of_each_col`
            (in one row: [c1, c2, c3, ...])
            
        c2:
            Get the "right" boundary,
            if and only if
            get the upper percentile of 
            `cusum_of_each_col`
            (in one row: [c1, c2, c3, ...])
        '''
        c1 = np.searchsorted(cusum_of_each_col,
                             np.percentile(cusum_of_each_col, lower_bound),
                             side="right")
        c2 = np.searchsorted(cusum_of_each_col,
                             np.percentile(cusum_of_each_col, upper_bound),
                             side="left")
        
        '''
        r1: 
            Get the "top" boundary,
            if and only if
            get the lower percentile of 
            `cusum_of_each_row` 
            (in one col: [r1, 
                          r2, 
                          r3, 
                          ...])
            
        r2:
            Get the "bottom" boundary,
            if and only if
            get the upper percentile of 
            `cusum_of_each_row`
            (in one col: [r1, 
                          r2, 
                          r3, 
                          ...])
        '''
        r1 = np.searchsorted(cusum_of_each_row,
                             np.percentile(cusum_of_each_row, lower_bound),
                             side="right")
        r2 = np.searchsorted(cusum_of_each_row,
                             np.percentile(cusum_of_each_row, upper_bound),
                             side="left")
        # If the method: 
        #    calculating "percentiles" would lead to get
        #    the wrong boundaries (`r1 > r2` or `c1 > c2`),
        # then use another method that calculate 
        #    proportion (in percentage) directly!
        if r1 > r2:
            h = img.shape[0]
            r1 = int(lower_bound/100 * h)
            r2 = int(upper_bound/100 * h)
        if c1 > c2:
            w = img.shape[1]
            c1 = int(lower_bound/100 * w)
            c2 = int(upper_bound/100 * w)

        # is_larger_crop: return both boundaries as the larger range
        if is_larger_crop:
            if (r2 - r1) > (c2 - c1):
                return [(r1, r2), (r1, r2)]
            else:
                return [(c1, c2), (c1, c2)]
        # default
        return [(r1, r2), (c1, c2)]
    
    def _compute_grid_points(self, img, n_grid_points=9, boundaries=None):
        """
        Returns:
            tuple of arrays indicating the vertical and horizontal locations of the grid points
        Examples:
            >>> img = gis.preprocess_img('https://pixabay.com/static/uploads/photo/2012/11/28/08/56/mona-lisa-67506_960_720.jpg')
            >>> boundaries = gis.crop_img(img)
            >>> gis.compute_grid_points(img, boundaries=boundaries)
            (array([100, 165, 230, 295, 360, 424, 489, 554, 619]),
             array([ 66, 109, 152, 195, 238, 280, 323, 366, 409]))
        """

        # if no boundaries are provided, use the entire image
        if boundaries is None:
            h, w = img.shape
            boundaries = [(0, h-1), (0, w-1)]
        r1, r2 = boundaries[0]
        c1, c2 = boundaries[1]
        x_coords = np.linspace(r1, r2, n_grid_points+2, dtype=int)[1:-1]
        y_coords = np.linspace(c1, c2, n_grid_points+2, dtype=int)[1:-1]
        #print("[test] length of x_coords:", len(x_coords))
        #print("[test] length of y_coords:", len(y_coords))
        return x_coords, y_coords
    
    def _compute_mean_level(self, img, x_coords, y_coords, P=None):
        if P is None:
            P = max([2.0, int(0.5 + min(img.shape)/20.)])     # per the paper

        avg_grey = np.zeros((x_coords.shape[0], y_coords.shape[0]))

        for i, x in enumerate(x_coords):        # not the fastest implementation
            lower_x_lim = int(max([x - P/2, 0]))
            upper_x_lim = int(min([lower_x_lim + P, img.shape[0]]))
            for j, y in enumerate(y_coords):
                lower_y_lim = int(max([y - P/2, 0]))
                upper_y_lim = int(min([lower_y_lim + P, img.shape[1]]))

                avg_grey[i, j] = np.mean(img[lower_x_lim:upper_x_lim,
                                        lower_y_lim:upper_y_lim])  # no smoothing here as in the paper

        return avg_grey
    
    def _compute_differentials(self, grey_level_matrix,  diagonal_neighbors=True):
        right_neighbors = -np.concatenate((np.diff(grey_level_matrix),
                                           np.zeros(grey_level_matrix.shape[0]).
                                           reshape((grey_level_matrix.shape[0], 1))),
                                          axis=1)
        left_neighbors = -np.concatenate((right_neighbors[:, -1:],
                                          right_neighbors[:, :-1]),
                                         axis=1)

        down_neighbors = -np.concatenate((np.diff(grey_level_matrix, axis=0),
                                          np.zeros(grey_level_matrix.shape[1]).
                                          reshape((1, grey_level_matrix.shape[1]))))

        up_neighbors = -np.concatenate((down_neighbors[-1:], down_neighbors[:-1]))

        if diagonal_neighbors:
            # this implementation will only work for a square (m x m) grid
            diagonals = np.arange(-grey_level_matrix.shape[0] + 1,
                                  grey_level_matrix.shape[0])

            upper_left_neighbors = sum(
                [np.diagflat(np.insert(np.diff(np.diag(grey_level_matrix, i)), 0, 0), i)
                 for i in diagonals])
            lower_right_neighbors = -np.pad(upper_left_neighbors[1:, 1:],
                                            (0, 1), mode='constant')

            # flip for anti-diagonal differences
            flipped = np.fliplr(grey_level_matrix)
            upper_right_neighbors = sum([np.diagflat(np.insert(
                np.diff(np.diag(flipped, i)), 0, 0), i) for i in diagonals])
            lower_left_neighbors = -np.pad(upper_right_neighbors[1:, 1:],
                                           (0, 1), mode='constant')

            return np.dstack(np.array([
                upper_left_neighbors,
                up_neighbors,
                np.fliplr(upper_right_neighbors),
                left_neighbors,
                right_neighbors,
                np.fliplr(lower_left_neighbors),
                down_neighbors,
                lower_right_neighbors]))

        return np.dstack(np.array([ up_neighbors,
                                    left_neighbors,
                                    right_neighbors,
                                    down_neighbors ]))
    
    def _normalize_and_threshold(self, difference_array,
                                 identical_tolerance=2/255., n_levels=2):
        # set very close values as equivalent
        mask = np.abs(difference_array) < identical_tolerance
        difference_array[mask] = 0.

        # if image is essentially featureless, exit here
        if np.all(mask):
            return None

        # bin so that size of bins on each side of zero are equivalent
        positive_cutoffs = np.percentile(difference_array[difference_array > 0.],
                                         np.linspace(0, 100, n_levels+1))
        negative_cutoffs = np.percentile(difference_array[difference_array < 0.],
                                         np.linspace(100, 0, n_levels+1))

        for level, interval in enumerate([positive_cutoffs[i:i+2]
                                          for i in range(positive_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array >= interval[0]) &
                             (difference_array <= interval[1])] = level + 1

        for level, interval in enumerate([negative_cutoffs[i:i+2]
                                          for i in range(negative_cutoffs.shape[0] - 1)]):
            difference_array[(difference_array <= interval[0]) &
                             (difference_array >= interval[1])] = -(level + 1)

        return None
    
    def calculate_signature(self, img_path):
        """ Step 1: Convert input RGB image to a grayscale """
        try:
            img = cv2.imread(img_path)
            
            # print(img.shape) # <-- (3d) height, width, channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(img.shape) # <-- (2d) height, width
            '''cv2.imshow("grayscale", img)
            cv2.waitKey(0)'''
    
    
            """ Step 2a: Determine cropping boundaries """
            img_boundaries = self._get_boundaries_of_cropped_image(img, self.lower_bound,
                                                                   self.upper_bound, self.is_larger_crop)
            
            """ Step 2b:   Generate grid centers """
            x_coords, y_coords = self._compute_grid_points(img, self.n_grid_points, img_boundaries)
    
            
            """ Step 3:    Compute grey level mean of each P x P
                           square centered at each grid point 
            """
            avg_grey = self._compute_mean_level(img, x_coords, y_coords, self.P)
            
            
            """ Step 4a:   Compute array of differences for each
                           grid point vis-a-vis each neighbor 
            """
            diff_mat = self._compute_differentials(avg_grey,
                                                   self.diagonal_neighbors)
            
            """ Step 4b: Bin differences to only 2n+1 values """
            self._normalize_and_threshold(diff_mat,
                                          self.identical_tolerance,
                                          self.n_levels)
    
            """ Step 5: Flatten array and return signature """
            return np.ravel(diff_mat).astype("int8")
        except:
            print("[ERR]", img_path)
            return None

    def calculate_normalized_distance(self, _a, _b):
        b = _b.astype(int)
        a = _a.astype(int)
        norm_diff = np.linalg.norm(b - a)
        norm1 = np.linalg.norm(b)
        norm2 = np.linalg.norm(a)
        return norm_diff / (norm1 + norm2)
    
    #---------------------------------------------------
    def get_difference(self, img_path_A, img_path_B):
        sigA = self.calculate_signature(img_path_A)
        sigB = self.calculate_signature(img_path_B)
        if all((sigA is not None, sigB is not None)):
            diff = self.calculate_normalized_distance(sigA, sigB)
            return diff
        else:
            return -1

if __name__ == "__main__":
    img_path_A = "images/紅色_下身類/00000000.png"
    img_path_B = "images/紅色_下身類/00000001.png"
    img_path_C = "images/紅色_下身類/00000002.jpg"
    
    img_path_D = "images/紅色_外套類/00000003.jpg"
    img_path_E = "images/紅色_外套類/00000004.jpg"
    img_path_F = "images/紅色_外套類/00000005.jpg"
    
    img_path_G = "images/godd/MonaLisa_WikiImages.jpg"
    img_path_H = "images/godd/MonaLisa_Wikipedia.jpg"
    img_path_I = "images/godd/MonaLisa_Remix_Flickr.jpg"
    
    img_path_J = "images/青色_襯衫類/00000006.jpg"
    img_path_K = "images/青色_襯衫類/00000007.jpg"
    img_path_L = "images/青色_襯衫類/00000008.jpg"
    
    img_path_S1 = "images/黑色_鞋類/00000009.jpg"
    img_path_S2 = "images/黑色_鞋類/00000010.jpg"
    img_path_S3 = "images/黑色_鞋類/00000011.jpg"
    
    img_path_X1 = "images/1.jpg"
    img_path_X2 = "images/2.jpg"
    img_path_X3 = "images/3.jpg"
    
    img_path_Y1 = "D:/MyPrograms/Python/py/專題/Cloth Image Classifier/dataset/img_db_3/000143_咖啡色_孕婦類/01010011 (41).jpg"
    
    phash = PHash()
    diff = phash.get_difference(img_path_A, img_path_A)
    print("diff(A, A):", diff)
    diff = phash.get_difference(img_path_A, img_path_B)
    print("diff(A, B):", diff)
    diff = phash.get_difference(img_path_A, img_path_C)
    print("diff(A, C):", diff)
    print()
    
    diff = phash.get_difference(img_path_A, img_path_D)
    print("diff(A, D):", diff)
    diff = phash.get_difference(img_path_A, img_path_E)
    print("diff(A, E):", diff)
    diff = phash.get_difference(img_path_A, img_path_F)
    print("diff(A, F):", diff)
    print()
    
    diff = phash.get_difference(img_path_A, img_path_G)
    print("diff(A, G):", diff)
    diff = phash.get_difference(img_path_A, img_path_H)
    print("diff(A, H):", diff)
    diff = phash.get_difference(img_path_A, img_path_I)
    print("diff(A, I):", diff)
    print()
    
    diff = phash.get_difference(img_path_A, img_path_J)
    print("diff(A, J):", diff)
    diff = phash.get_difference(img_path_A, img_path_K)
    print("diff(A, K):", diff)
    diff = phash.get_difference(img_path_A, img_path_L)
    print("diff(A, L):", diff)
    print()
    
    diff = phash.get_difference(img_path_A, img_path_S1)
    print("diff(A, S1):", diff)
    diff = phash.get_difference(img_path_A, img_path_S2)
    print("diff(A, S2):", diff)
    diff = phash.get_difference(img_path_A, img_path_S3)
    print("diff(A, S3):", diff)
    print()
    
    diff = phash.get_difference(img_path_A, img_path_X1)
    print("diff(A, X1):", diff)
    diff = phash.get_difference(img_path_A, img_path_X2)
    print("diff(A, X2):", diff)
    diff = phash.get_difference(img_path_A, img_path_X3)
    print("diff(A, X3):", diff)
    print()
    
    diff = phash.get_difference(img_path_A, img_path_Y1)
    print("diff(A, Y1):", diff)
