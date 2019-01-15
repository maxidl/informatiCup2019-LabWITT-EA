"""
This file is a stand-alone script to analyze each class of the target API. It queries the entire
test set and saves the response to file. Each class will be analyzed with respect to the
classification accuracy. In addition, a specific class of the training set can be analyzed.

"""


from io import BytesIO
import numpy as np
import pandas as pd
import time, os, csv, requests
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array


URL = "<insert api url"
API_KEY = "<insert api key>"

def process_image(img, img_size):
    """
    Loads an image from file, resize to img_size and return as numpy array
    :param img: image path
    :param img_size: image size as tuple (width, height)
    :return: ndarray (width, height, 3)
    """
    img = load_img(img, target_size=img_size)
    img = img_to_array(img)
    img = img/255
    img = img.clip(0, 1)
    return img

def send_query(image):
    """
    Arguments:
        image {[np array, shape (X, Y, 3)]} -- [numpy array of the image to be send]
    Returns:
        ndarray of sorted labels and confidence
    """

    # create file object from image
    image_file = BytesIO()
    if image.shape[2] == 1:
        image = np.squeeze(image)
        plt.imsave(image_file, image)
    else:
        plt.imsave(image_file, image, cmap='gray')
    image_file.seek(0)

    # send post requests
    response = requests.post(
        URL, params={'key': API_KEY}, files={'image': image_file})

    while response.status_code == 429 or response.status_code == 400:
        print("query limit reached, waiting 60s...")
        time.sleep(60)
        return send_query(image)
    # process response
    if response.status_code == 200:  # OK
        predictions = response.json()
        df = pd.DataFrame(predictions)
        return df['class'].values, df.confidence.values
    else:
        print(response.status_code)
        print(response.content)
        raise ConnectionError


def get_images_from_annotation_file(annotation_path, img_path, images_per_class=None):
    """
    Loads the test set with labels specified in a csv. If images_per_class set, only a subset will be selected.

    :param annotation_path: Path to 'GT-final_test.csv'
    :type annotation_path: str
    :param img_path: Path to test images
    :type img_path: str
    :param images_per_class: Limit for images per class
    :type images_per_class: int
    :return: List of all selected images and a DataFrame which contain all information
    :rtype: (np.ndarray, pd.DataFrame)
    """

    df = pd.read_csv(annotation_path, sep=";")
    df = df.drop(columns=["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"])

    annotations = pd.DataFrame(None, columns=df.columns)
    for i in range(np.unique(df.ClassId.values).size):
        sub_df = df.loc[df.ClassId == i]

        if images_per_class and images_per_class > 0 and images_per_class < df.loc[df.ClassId == i].ClassId.size:
            sub_df = sub_df[:images_per_class]
        annotations = annotations.append(sub_df, ignore_index=True)

    xtest = np.zeros((annotations.Filename.size, 64, 64, 3))
    for i, row in annotations.iterrows():
        xtest[i] = process_image(os.path.join(img_path, row.Filename), (64, 64))
    return xtest, annotations


def analyze_api(img_path, annotations_path, file_out_path, images_per_class=None):
    if images_per_class:
        X, annotations = get_images_from_annotation_file(annotations_path, img_path, images_per_class)
    else:
        # query all test images
        X, annotations = get_images_from_annotation_file(annotations_path, img_path)

    # save results batch wise to csv
    columns = ["id", "gt_class_id", "img_name", "top-1_p", "top-2_p", "top-3_p", "top-4_p", "top-5_p", "top-1_l",
               "top-2_l", "top-3_l", "top-4_l", "top-5_l"]
    dff = pd.DataFrame(columns=columns)
    dff.to_csv(file_out_path, encoding="utf8", index=False)

    for i, row in annotations.iterrows():
        labels, probs = send_query(X[i])
        time.sleep(1)

        csv_row = [i, row["ClassId"], row["Filename"]]

        for j in range(5):
            csv_row.append(probs[j])
        for j in range(5):
            csv_row.append(labels[j])

        print("predicton of class {}: {} ({}) ({}/{})".format(row.ClassId, labels[0], probs[0], i,
                                                              len(annotations.index)))
        dff.loc[i] = csv_row

        if i % 60 == 0 and i > 0:
            with open(file_out_path, "a") as f_out:
                dff.to_csv(f_out, encoding="utf8", header=False, index=False)
            dff = dff[0:0]  # drop current data

    with open(file_out_path, "a") as f_out:
        dff.to_csv(f_out, encoding="utf8", header=False, index=False)


def analyze_results(file_out_path, img_path, out_path_images):
    df = pd.read_csv(file_out_path)

    print("unique top-1 labels:")
    print(df["top-1_l"].unique())
    print(len(df["top-1_l"].unique()))

    unique_labels = [df["top-1_l"].unique().tolist(),
                     df["top-2_l"].unique().tolist(),
                     df["top-3_l"].unique().tolist(),
                     df["top-4_l"].unique().tolist(),
                     df["top-5_l"].unique().tolist()
                     ]
    unique_labels = set([item for sublist in unique_labels for item in sublist])
    print("unique top-5 labels")
    print(list(unique_labels))
    print(len(unique_labels))

    # missed labels in top-1 list
    print("labels missed in top-1 prediction:")
    print(set(df["top-1_l"]).symmetric_difference(unique_labels))

    for i in range(df["gt_class_id"].unique().size):
        # for each class make the prediction analysis
        class_rows = df.loc[df["gt_class_id"] == i]

        # prediction distribution
        # print(class_rows["top-1_l"].value_counts())

        # get the most frequent predicted label
        label = class_rows["top-1_l"].value_counts().idxmax()

        # get the image with the highest prediction for the most frequent label
        class_rows = class_rows.loc[df["top-1_l"] == label]
        best_prediction = class_rows.loc[class_rows["top-1_p"].idxmax()]

        # save this image to file for validation
        img = io.imread(os.path.join(img_path, best_prediction["img_name"]))

        if not os.path.exists(out_path_images):
            os.makedirs(out_path_images)
        io.imsave(
            os.path.join(out_path_images,
                         str(i) + "_" + str(best_prediction["top-1_p"]) + "_" + best_prediction[
                             "top-1_l"] + ".png"),
            img)


def measure_accuracy(file_out_path, index_label_dict_path):

    df = pd.read_csv(file_out_path)

    # accuracy measured by the gtsrb testset
    infile = open(os.path.join(index_label_dict_path), mode="r")
    reader = csv.reader(infile)
    index_label_dict = {rows[0]: rows[1] for rows in reader}
    label_index_dict = {v: k for k, v in index_label_dict.items()}

    for i, row in df.iterrows():
        df.at[i, "top-1_l"] = label_index_dict[row["top-1_l"]]

    from sklearn.metrics import accuracy_score
    y_true = np.array(df["gt_class_id"].tolist()).astype(int)
    y_pred = np.array(df["top-1_l"].tolist()).astype(int)

    print("overall accuracy:")
    print(accuracy_score(y_true, y_pred, normalize=True))

    from sklearn.metrics import confusion_matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # normalize
    cnf_matrix = cnf_matrix / cnf_matrix.astype(np.float).sum(axis=1)

    # accuracy for each class
    print("accuracy for each class:")
    for i in range(len(cnf_matrix)):
        for j in range(len(cnf_matrix[0])):
            if i == j:
                print("true: {}, acc:{}".format(i, cnf_matrix[i, j]))


def analyze_class(class_id, training_path_class, result_file, limit=None):

    # collect filenames
    file_names = [f for f in os.listdir(training_path_class) if f.endswith(".ppm")]

    # select a subset randomly
    if limit > 0 and limit < len(file_names):
        np.random.shuffle(file_names)
        file_names = file_names[:limit]

    # load these images from file
    query_images = np.zeros((len(file_names), 64, 64, 3))
    for i in range(len(query_images)):
        query_images[i] = process_image(os.path.join(training_path_class, file_names[i]), (64, 64))


    # analyze each image and in addition save results to file
    columns = ["id", "gt_class_id", "img_name", "top-1_p", "top-2_p", "top-3_p", "top-4_p", "top-5_p", "top-1_l",
               "top-2_l", "top-3_l", "top-4_l", "top-5_l"]
    df = pd.DataFrame(columns=columns)
    df.to_csv(result_file, encoding="utf8", index=False)

    for i in range(len(query_images)):
        labels, probs = send_query(query_images[i])
        time.sleep(1)

        row = [i, class_id, file_names[i]]
        for j in range(5):
            row.append(probs[j])
        for j in range(5):
            row.append(labels[j])

        print("{} ({}) ({}/{})".format(labels[0], probs[0], i+1, len(query_images)))
        df.loc[i] = row

    with open(result_file, "a") as f_out:
        df.to_csv(f_out, encoding="utf8", header=False, index=False)

    print("top-1 labels:")
    print(df["top-1_l"].value_counts())

    return


def main():

    file_out_path = "res/test_images_analysis_full.csv"
    annotations_path = "res/GTSRB/test/Final_Test/GT-final_test.csv"

    # query the entire test set and save the response to file
    img_path = "res/GTSRB/test/Final_Test/Images"
    # analyze_api(img_path, annotations_path, file_out_path)


    # 1. list unique top-1 labels
    # 2. list all unique labels in top-5
    # 3. save the image with the highest score of the most frequent labels to file
    out_path_images = "res/test_img_val"
    analyze_results(file_out_path, img_path, out_path_images)

    # manually build a index to label mapping using these images

    # measure the overall accuracy and the accuracy for each class
    index_label_dict_path = "res/index_label_dict.csv"
    measure_accuracy(file_out_path, index_label_dict_path)


    # analyze one specific class on training set
    training_path_class = "res/GTSRB/train/Final_Training/Images/00041"
    class_id = 41
    result_file = "results/train_analysis_41.csv"
    analyze_class(class_id, training_path_class, result_file, limit=60)


if __name__ == "__main__":
    main()
