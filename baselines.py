import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

class BenchmarkBaselines:
    def __init__(self, X, y, k, lambda_weight = 1.0):
        self.answer_to_the_universe = 42

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size = 0.3, random_state = self.answer_to_the_universe
        )

        self.k = k #For top k selection and some others like random
        self.n_features = X.shape[1]
        self.lambda_weight = lambda_weight

        self.relevance = mutual_info_classif(self.X_train, self.y_train, random_state=self.answer_to_the_universe)
        self.redundancy = self.compute_redundancy_matrix()

    def compute_redundancy_matrix(self):
        """
        Computes the redundancy matrix R used in QUBO
        I need this so my greedy algo knows what to be greedy towards
        It's O(n^2) but it's fine cause it runs once only in the init
        """
        #NOTE regression assumes our dataset is continuous data, for the breast cancer set this is true, but if not I can add a if-else to deal with discrete using a classifier
        #Calculate pairwise mutual info and stuff it into the matrix
        matrix = np.zeros((self.n_features, self.n_features))
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                mi_ij = mutual_info_regression(self.X_train[:, [i]], self.X_train[:, j], random_state=self.answer_to_the_universe)
                #mutual_info_regression returns some funky array which numpy won't unpack for me
                matrix[i,j] = mi_ij[0]
                matrix[j,i] = mi_ij[0]

        return matrix

    def evaluate(self, features):
        """
        Uniform evaluate function for all the models
        Just plug your final selected feature indices into the function to get a score
        """
        if not features:
            return 0.0
        classif = RandomForestClassifier(random_state=self.answer_to_the_universe)
        classif.fit(self.X_train[:, features], self.y_train)
        pred = classif.predict(self.X_test[:, features])
        return accuracy_score(self.y_test, pred)


    """
    The idea for the following functions is that they select column indices
    from our chosen data set X
    """

    def select_all_features(self):
        return list(range(self.n_features))

    def select_random(self, seed = 42):
        np.random.seed(seed)
        return list(np.random.choice(self.n_features, self.k, replace = False))

    def select_top_k(self):
        #Pre-calc'd relevance in init so we can just sort it big to small and select k of them
        return list(np.argsort(self.relevance) [-self.k:])

    def select_greedy(self):
        # MAde it super close to how a QUBO QAOA is selecting
        # In essense, this is optimising for the same things, but local only since it selects next best thing each time
        # The QAOA sees the whole global field at once with it's beeg brain
        selected = []
        frontier = list(range(self.n_features))

        #almost forgot this lol, initial test looked real dumdum withotu a starting feature
        first_best = np.argmax(self.relevance)
        selected.append(first_best)
        frontier.remove(first_best)

        for i in range(self.k - 1):
            best_score = -float('inf')
            best_feat = None

            for feature in frontier:
                avg_redundancy = np.mean([self.redundancy[feature, s] for s in selected])
                score = self.relevance[feature] - (self.lambda_weight * avg_redundancy)

                if score > best_score:
                    best_score = score
                    best_feat = feature

            selected.append(best_feat)
            frontier.remove(best_feat)

        return selected


# Example of usage
if __name__ == "__main__":

    # Dummy data for testing
    # X, y = make_classification(
    #     n_samples=1000, n_features=25, n_informative=8, n_redundant=8, random_state=42
    # )

    X, y = load_breast_cancer(return_X_y=True)
    #NOTE If you use this dataset alone, because it is super clean already, feature selection actually underperforms just selecting everything
    # I'm adding some noise to make it not so clean just for testing to make sure I'm actually selecting some features

    """
    The idea is Random Forest is my evaluation, Random Forest is inherently able to feature select so I try to overwhelm is with 
    a crazy amount of noise, from  my tests at 500 noise top K now outperforms seelcting all and Greedy ties with it

    Realistically for our project, this doesn't matter, ideally the QAOA will outperform the baselines and our goal is proven anyways
    The breast cancer dataset is very clean and has some super correlated data points that should not be feature selected against anyways
    It should be the case that any form of feature selection will lose to select all on the raw dataset itself from what I can tell
    """
    NUM_NOISE = 500 #Set to 0 if you don't want noise and want this program to run at a reasonable time
    np.random.seed(42) #Seed again just in case you ran select random earlier which changes the seed too
    noise = np.random.randn(X.shape[0], NUM_NOISE) #Create random noise in shape of dataset
    X_noisy = np.hstack((X, noise)) #Glue noise to dataset


    benchmark = BenchmarkBaselines(X_noisy, y, k=5, lambda_weight=1.0)

    feats_all = benchmark.select_all_features()
    feats_rand = benchmark.select_random()
    feats_topk = benchmark.select_top_k()
    feats_greedy = benchmark.select_greedy()

    score_all = benchmark.evaluate(feats_all)
    score_rand = benchmark.evaluate(feats_rand)
    score_topk = benchmark.evaluate(feats_topk)
    score_greedy = benchmark.evaluate(feats_greedy)

    print("Score (All): ", score_all)
    print("Score (Random): ", score_rand)
    print("Score (Top-K): ", score_topk)
    print("Score (Greedy): ", score_greedy)