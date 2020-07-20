from sklearn.tree import DecisionTreeRegressor
from data_generator import *
import matplotlib.pyplot as plt


class LambdaMART:

    def __init__(self, training_data=None, number_of_trees=5, learning_rate=0.5, tree_type='sklearn'):
        """
            This is the constructor for the LambdaMART object.
            Parameters
            ----------
            training_data : list of int
            Contain a list of numbers
            number_of_trees : int (default: 5)
            Number of trees LambdaMART goes through
            learning_rate : float (default: 0.1)
            Rate at which we update our prediction with each tree
            tree_type : string (default: "sklearn")
            Either "sklearn" for using Sklearn implementation of the tree of "original" 
            for using our implementation
        """

        # if tree_type != 'sklearn' and tree_type != 'original':
        # 	raise ValueError('The "tree_type" must be "sklearn" or "original"')
        self.training_data = training_data
        self.number_of_trees = number_of_trees
        self.learning_rate = learning_rate
        self.trees = []
        self.tree_type = tree_type
        self.X_train, self.X_test, self.y_train, self.y_test = generate_data()
    
    def train(self):
        """
            fit first tree with training data
            fit other trees with residual
        """
        if self.tree_type == 'sklearn':
            # build base tree
            tree = DecisionTreeRegressor(max_depth=5)
            tree.fit(self.X_train, self.y_train)
            self.trees.append(tree)
            
            # build other trees in residual
            for _ in range(self.number_of_trees - 1):
                # Sklearn implementation of the tree
                prediction = self.predict(self.X_train)
                residual = self.y_train - prediction
                            
                tree = DecisionTreeRegressor(max_depth=3)
                tree.fit(self.X_train, residual)
                self.trees.append(tree)

    def predict(self, data):
        """
        accumulate prediction through each tree by learning rate
        """
        prediction = np.zeros(data.shape[0])
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(data)
        return prediction
    
    def test(self, data):
        """
        accumulate prediction through each tree by learning rate
        """
        prediction = np.zeros(data.shape[0])
        for tree in self.trees:
            prediction += self.learning_rate * tree.predict(data)
            yield prediction
    
if __name__ == '__main__':
    l = LambdaMART()
    l.train()
    
    p = plt.figure()
    idx = 1
    for prediction in l.test(l.X_test):
        ax = p.add_subplot(3,3,idx)
        ax.plot(prediction, 'b', label='prediction')
        ax.plot(l.y_test, 'r', label='groundtruth')
        idx += 1
    plt.show()