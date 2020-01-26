"""Making functions for saving and loading files"""


def save_object(object_to_save, filename="filename"):
    """Saves an object with the filename: "filename.sav" in the current folder"""
    import joblib

    object_to_save = object_to_save
    object_saved_as = f"{filename}.sav"
    joblib.dump(object_to_save, object_saved_as)


def load_object(load_file):
    """Load object from its location and return object"""
    import joblib

    return joblib.load(load_file)


"""END Making functions for save and load"""
