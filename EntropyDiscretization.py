import numpy as np
import math

class EntropyDiscretization:
    def __init__(self):
        self.num_classes = 0
        self.midpoints = []

    
    def P(self, class_partition, C):
        """Returns the proprotion of examples in S (class_partition) that have the Class i"""
        count = np.count_nonzero(class_partition == C)
        if class_partition.size == 0:
            proportion = 0
        else:
            proportion = count / class_partition.size
        proportion += 1
        #print(proportion)
        return proportion
    
    def entropy(self, class_partition):
        """Returns the class entropy which is used in the definition of class entropy"""
        ent = 0
        for C in self.classes:
            ent +=  -1 * self.P(class_partition, C) * math.log(self.P(class_partition, C))
        #ent = -1 * sum([self.P(class_partition, C) * math.log(self.P(class_partition, C)) for C in self.classes])
        return ent
    
    def information_entropy(self, class_partition1, class_partition2):
        """Returns the class information entropy which is used to determine the right cutpoint"""
        info_entropy = ((class_partition1.size / self.size) * self.entropy(class_partition1)) + ((class_partition2.size / self.size) * self.entropy(class_partition2))
        return info_entropy
    
    def binarize(self, attr_x, y):
        """Binarize a particular attribute based on the most minimal class entropy"""
        sorted_x, sorted_y = self.sort(attr_x, y)
        cut_points = np.where(sorted_y[:-1] != sorted_y[1:], (sorted_x[:-1] + sorted_x[1:])/2, np.nan)
        best_cut = 0
        info_entropy = np.inf
        cut_points = cut_points[~np.isnan(cut_points)]
        for cut in cut_points:
            tmp_entropy = self.information_entropy(sorted_y[sorted_x <= cut], sorted_y[sorted_x > cut])
            if info_entropy > tmp_entropy:
                info_entropy = tmp_entropy
                best_cut = cut
        #print("best cut:", best_cut)
        return best_cut
    
    def fit(self, x, y):
        self.classes = np.unique(y)
        self.num_classes = self.classes.size
        self.size = x.size
        self.size, columns = x.shape
        # self.size, columns = x.shape
        for i in range(0, columns):
            self.midpoints.append(self.binarize(x[:,i], y))
    
    def fit_transform(self, x, y):
        self.fit( x, y)
        return self.transform(x)
    
    def transform(self, x):
        _, columns = x.shape
        if columns != len(self.midpoints):
            raise Exception('Size of x columns is not equal to the number of midpoints')
        return np.where(x[:,:] > self.midpoints, 1, 0)
    
        
    
    def sort(self, attr_x, y):
        """Returns attr_x and y sorted maintaining the relationship between the two
        Returns: sorted_x, sorted_y"""
        arrinds = np.argsort(attr_x)
        x_sorted = attr_x[arrinds[::]]
        y_sorted = y[arrinds[::]].flatten()
        return x_sorted, y_sorted
    