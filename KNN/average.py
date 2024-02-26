from scipy.sparse import csr_matrix
import numpy as np

class AverageRating:
    def __init__(self):
        self.predictions = None
        self.training_shape = None

    def fit(self, training_ratings, withZeros=False):
        predictions = None
        if withZeros:
            predictions = training_ratings.mean(axis=0)
        else:
            sums = training_ratings.sum(axis=0)
            nnz = training_ratings.getnnz(axis=0)
            nnz = [1 if n == 0 else n for n in nnz] # make sure we aren't dividing by zero
            predictions = sums / nnz
        predictions = csr_matrix(predictions, dtype=np.float32)
        self.predictions = predictions
        
    def predict(self, user_ratings):
        if self.predictions is None:
            raise Exception("Must fit before predicting")
        prediction_matrix = self.__expand_csr(self.predictions, user_ratings.shape[0])
        return prediction_matrix
    
    def __expand_csr(self, csr, n_rows):
        """
        Expands a csr matrix that is a row vector into a matrix.
        It duplicates the row passed in n_rows times.
        """
        
        new_data = np.tile(csr.data, n_rows)
        new_indices = np.tile(csr.indices, n_rows)
        indptr = csr.indptr
        row_length = indptr[1]
        new_indptr = np.linspace(0,row_length*n_rows, n_rows + 1)
        return csr_matrix((new_data.astype(np.float32), new_indices.astype(np.int32), new_indptr.astype(np.int32)), dtype=np.float32)