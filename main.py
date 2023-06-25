import numpy as np
from sklearn.model_selection import train_test_split
from model import model_KNN
def main(file):
    # Step 1: Load the entire dataset from the file
    if file == "Caesarian.txt" :
         data = np.loadtxt(file, delimiter=',')
    else :
        data = np.loadtxt(file, delimiter=' ')

    # Split the dataset into features (X) and labels (y)
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels

    # Step 2: Split the dataset into training and test sets


    # Define the values of k and p
    k_values = [1, 3, 5, 7, 9]
    p_values = [1, 2, np.inf]

    imprial_error_table = {}
    # Initialize the error table for imperial error
    for k in k_values:
        for p in p_values:
            imprial_error_table[f"{p},{k}"] = 0

    true_error_table = {}
    # Initialize the error table for true error
    for k in k_values:
        for p in p_values:
            true_error_table[f"{p},{k}"] = 0


    for i in range(0, 100):
        # new disterbution :
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    # Step 3 and 4: Implement and evaluate the k-NN classifier for different k and p values
        for k in k_values:
            for p in p_values:
                # Create and fit the k-NN classifier
                model = model_KNN(k, p, X_train, y_train,  X_train, y_train)
                predictions = model.train()
                imprial_error = 1 - (np.sum(y_train == predictions) / len(y_train))
                imprial_error_table[f"{p},{k}"] += imprial_error

                # Evaluate the classifier on the test set
                model = model_KNN(k, p, X_train, y_train,  X_test, y_test)
                predictions = model.train()
                test_error = 1 - (np.sum(y_test == predictions) / len(y_test))
                true_error_table[f"{p},{k}"] += test_error



    for x, sum_errors in imprial_error_table.items():
        print(f"{x} : {round(sum_errors / 100.0, 4):.4f}")

    for x, sum_errors in true_error_table.items():
        print(f"{x} : {round(sum_errors / 100.0, 4):.4f}")


if __name__ == '__main__':
    print("###################################### Caesarian  ############################# ")
    main("Caesarian.txt")
    print("###################################### two_circle  ############################# ")
    main("two_circle.txt")
