import random
from sklearn.model_selection import train_test_split

def random_split_paths(paths, validation_ratio=0.2, random_seed=None):
    """
    Randomly splits a list of paths into training and validation sets.

    Args:
        paths (list): List of file paths.
        validation_ratio (float): Proportion of data to allocate for validation.
        random_seed (int): Seed for randomization.

    Returns:
        tuple: Lists of train and validation paths.
    """
    random.seed(random_seed)
    
    train_paths, validation_paths = train_test_split(paths, test_size=validation_ratio, random_state=random_seed)
    
    return train_paths, validation_paths