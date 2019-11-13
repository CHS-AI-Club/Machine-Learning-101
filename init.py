try:
    import numpy as np
    import pandas as pd
except (ModuleNotFoundError, ImportError):
    print("You are probably not running this file on the Anaconda environment; file couldn't import numpy or pandas.")
    exit()
import os, sys

VELOCITY = 335
ACCELERATION = -9.81
SAMPLES = 40 
STANDARD_DEVIATION = 0.005 

def initialize():
    # data initialization
    np.random.seed(0)
    x = np.linspace(start=0.0, stop=200, num=SAMPLES, endpoint=False)
    t = x / VELOCITY
    rand = np.random.normal(loc=0.0, scale=STANDARD_DEVIATION, size=[SAMPLES])
    y = 0.5 * ACCELERATION * (t ** 2) + rand
    dataset = np.transpose([x, y])
    dataset = pd.DataFrame(dataset)
    dataset.columns = ['distance', 'bullet_drop']
    train_set = dataset.loc[dataset['distance'] <= 100]
    test_set = dataset.loc[dataset['distance'] > 100]

    # creating files 
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_101")
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass
    train_set.to_csv(os.path.join(directory, "train_set.csv"))
    test_set.to_csv(os.path.join(directory, "test_set.csv"))
    init_program_file(directory)
    self_destroy()

def init_program_file(directory):
    main_path = os.path.join(directory, "main.py")
    with open(main_path, 'w') as f:
        f.write("\n''' Follow the instructions of the comments and put the code blow the lines.")
        f.write("\nif you need help, ask around and check with the officers. '''")
        f.write("\n\n# import necessary packages")
        f.write("\n\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt")
        f.write("\nfrom sklearn.preprocessing import PolynomialFeatures")
        f.write("\nfrom sklearn.linear_model import LinearRegression")
        f.write("\nfrom sklearn.pipeline import make_pipeline")
        f.write("\nimport pickle")
        f.write("\n\n# read in the train_set.csv")
        f.write("\n\n\n\n# pull the x and y from the dataset and convert them to np arrays")
        f.write("\n\n\n\n# visualize the data using simple scatter plot")
        f.write("\n\n\n\n# make the model and fit the data into it")
        f.write("\n\n\n\n# use the model to predict the bullet drop at 150 meters")
        f.write("\n\n\n\n# fit the regression curve into the original scatter plot")    

def self_destroy():
    os.remove(sys.argv[0])

if __name__ == "__main__":
    initialize()
