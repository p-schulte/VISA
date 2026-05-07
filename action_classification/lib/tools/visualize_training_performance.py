import re
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_loader import CONFIG

LOG_ROOT = os.environ.get("VISA_LOG_ROOT", "logs")

INPUT_FILE= "a.txt"
OUTPUT_FILE= "training_loss.png"

def plot_args_seperately():

    # Loading the configuration and then obtain file-paths
    conf = CONFIG.dsg
    DATASET_NAME = conf['DATASET_NAME']
    try:
        RUN_NAME = '/' + CONFIG.ac['run_name']
    except:
        RUN_NAME = ""
    INPUT_FILE = os.path.join(LOG_ROOT, "ac_train", f"{DATASET_NAME}{RUN_NAME}", "run.txt")
    OUTPUT_FILE_BASE = os.path.join(LOG_ROOT, "ac_train", f"{DATASET_NAME}{RUN_NAME}")

    # Load the log file
    file_path = INPUT_FILE
    with open(file_path, "r") as file:
        log_data = file.readlines()

    # Regex patterns to extract relevant information
    epoch_pattern = re.compile(r"Evaluation after epoch (\d+)")
    train_pattern = re.compile(
        r"Train set: \[Loss Combined = ([\d\.]+), Loss Action Name = ([\d\.]+), Loss Argument 1 = ([\d\.]+), Loss Argument 2 = ([\d\.]+), Accuracy Combined = ([\d\.]+), Accuracy Action Name = ([\d\.]+), Accuracy Argument 1 = ([\d\.]+), Accuracy Argument 2 = ([\d\.]+)\]"
    )
    test_pattern = re.compile(
        r"Test set: \[Loss Combined = ([\d\.]+), Loss Action Name = ([\d\.]+), Loss Argument 1 = ([\d\.]+), Loss Argument 2 = ([\d\.]+), Accuracy Combined = ([\d\.]+), Accuracy Action Name = ([\d\.]+), Accuracy Argument 1 = ([\d\.]+), Accuracy Argument 2 = ([\d\.]+)\]"
    )

    # Lists to store extracted data
    epochs = []
    train_loss_action_name, train_loss_arg1, train_loss_arg2 = [], [], []
    train_acc_action_name, train_acc_arg1, train_acc_arg2 = [], [], []
    test_loss_action_name, test_loss_arg1, test_loss_arg2 = [], [], []
    test_acc_action_name, test_acc_arg1, test_acc_arg2 = [], [], []

    # Extract data from log file
    for line in log_data:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)

        train_match = train_pattern.search(line)
        if train_match:
            train_loss_action_name.append(float(train_match.group(2)))
            train_loss_arg1.append(float(train_match.group(3)))  # Use Loss Argument 1 directly
            train_loss_arg2.append(float(train_match.group(4)))  # Use Loss Argument 2 directly
            train_acc_action_name.append(float(train_match.group(6)))
            train_acc_arg1.append(float(train_match.group(7)))  # Use Accuracy Argument 1 directly
            train_acc_arg2.append(float(train_match.group(8)))  # Use Accuracy Argument 2 directly

        test_match = test_pattern.search(line)
        if test_match:
            test_loss_action_name.append(float(test_match.group(2)))
            test_loss_arg1.append(float(test_match.group(3)))  # Use Loss Argument 1 directly
            test_loss_arg2.append(float(test_match.group(4)))  # Use Loss Argument 2 directly
            test_acc_action_name.append(float(test_match.group(6)))
            test_acc_arg1.append(float(test_match.group(7)))  # Use Accuracy Argument 1 directly
            test_acc_arg2.append(float(test_match.group(8)))  # Use Accuracy Argument 2 directly

    # Ensure all lists have the same length
    min_length = min(
        len(epochs),
        len(train_loss_action_name),
        len(train_loss_arg1),
        len(train_loss_arg2),
        len(train_acc_action_name),
        len(train_acc_arg1),
        len(train_acc_arg2),
        len(test_loss_action_name),
        len(test_loss_arg1),
        len(test_loss_arg2),
        len(test_acc_action_name),
        len(test_acc_arg1),
        len(test_acc_arg2),
    )
    epochs = epochs[:min_length]
    train_loss_action_name = train_loss_action_name[:min_length]
    train_loss_arg1 = train_loss_arg1[:min_length]
    train_loss_arg2 = train_loss_arg2[:min_length]
    train_acc_action_name = train_acc_action_name[:min_length]
    train_acc_arg1 = train_acc_arg1[:min_length]
    train_acc_arg2 = train_acc_arg2[:min_length]
    test_loss_action_name = test_loss_action_name[:min_length]
    test_loss_arg1 = test_loss_arg1[:min_length]
    test_loss_arg2 = test_loss_arg2[:min_length]
    test_acc_action_name = test_acc_action_name[:min_length]
    test_acc_arg1 = test_acc_arg1[:min_length]
    test_acc_arg2 = test_acc_arg2[:min_length]

    # Create DataFrame
    df = pd.DataFrame({
        "Epoch": epochs,
        "Train Loss Action Name": train_loss_action_name,
        "Train Loss Arg1": train_loss_arg1,
        "Train Loss Arg2": train_loss_arg2,
        "Train Accuracy Action Name": train_acc_action_name,
        "Train Accuracy Arg1": train_acc_arg1,
        "Train Accuracy Arg2": train_acc_arg2,
        "Test Loss Action Name": test_loss_action_name,
        "Test Loss Arg1": test_loss_arg1,
        "Test Loss Arg2": test_loss_arg2,
        "Test Accuracy Action Name": test_acc_action_name,
        "Test Accuracy Arg1": test_acc_arg1,
        "Test Accuracy Arg2": test_acc_arg2,
    })

        # Create a figure with three subplots for Loss
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Plot Loss for Action Name
    axes[0].plot(df["Epoch"], df["Train Loss Action Name"], label="Train Loss Action Name", marker="o")
    axes[0].plot(df["Epoch"], df["Test Loss Action Name"], label="Test Loss Action Name", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss for Action Name over Epochs")
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].legend()

    # Plot Loss for Arg1
    axes[1].plot(df["Epoch"], df["Train Loss Arg1"], label="Train Loss Arg1", marker="o")
    axes[1].plot(df["Epoch"], df["Test Loss Arg1"], label="Test Loss Arg1", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss for Arg1 over Epochs")
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].legend()

    # Plot Loss for Arg2
    axes[2].plot(df["Epoch"], df["Train Loss Arg2"], label="Train Loss Arg2", marker="o")
    axes[2].plot(df["Epoch"], df["Test Loss Arg2"], label="Test Loss Arg2", marker="s")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Loss for Arg2 over Epochs")
    axes[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[2].legend()

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_BASE + "loss_combined.png")
    print(f"Saved combined loss plot to '{OUTPUT_FILE_BASE}loss_combined.png'.")

    # Create a figure with three subplots for Accuracy
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Plot Accuracy for Action Name
    axes[0].plot(df["Epoch"], df["Train Accuracy Action Name"], label="Train Accuracy Action Name", marker="o")
    axes[0].plot(df["Epoch"], df["Test Accuracy Action Name"], label="Test Accuracy Action Name", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy for Action Name over Epochs")
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].legend()

    # Plot Accuracy for Arg1
    axes[1].plot(df["Epoch"], df["Train Accuracy Arg1"], label="Train Accuracy Arg1", marker="o")
    axes[1].plot(df["Epoch"], df["Test Accuracy Arg1"], label="Test Accuracy Arg1", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy for Arg1 over Epochs")
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].legend()

    # Plot Accuracy for Arg2
    axes[2].plot(df["Epoch"], df["Train Accuracy Arg2"], label="Train Accuracy Arg2", marker="o")
    axes[2].plot(df["Epoch"], df["Test Accuracy Arg2"], label="Test Accuracy Arg2", marker="s")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Accuracy for Arg2 over Epochs")
    axes[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[2].legend()

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_BASE + "accuracy_combined.png")
    print(f"Saved combined accuracy plot to '{OUTPUT_FILE_BASE}accuracy_combined.png'.")



def plot_args_combined():

    # Loading the configuration and then obtain file-paths
    conf = CONFIG.dsg
    DATASET_NAME = conf['DATASET_NAME']
    try:
        RUN_NAME = '/' + CONFIG.ac['run_name']
    except:
        RUN_NAME = ""
    INPUT_FILE = os.path.join(LOG_ROOT, "ac_train", f"{DATASET_NAME}{RUN_NAME}", "run.txt")
    OUTPUT_FILE_BASE = os.path.join(LOG_ROOT, "ac_train", f"{DATASET_NAME}{RUN_NAME}")

    # Load the log file
    file_path = INPUT_FILE
    with open(file_path, "r") as file:
        log_data = file.readlines()

    # Regex patterns to extract relevant information
    epoch_pattern = re.compile(r"Evaluation after epoch (\d+)")
    train_pattern = re.compile(
        r"Train set: \[Loss Combined = ([\d\.]+), Loss Action Name = ([\d\.]+), Loss Arguments = ([\d\.]+), Accuracy Combined = ([\d\.]+), Accuracy Action Name = ([\d\.]+), Accuracy Arguments = ([\d\.]+)\]"
    )
    test_pattern = re.compile(
        r"Test set: \[Loss Combined = ([\d\.]+), Loss Action Name = ([\d\.]+), Loss Arguments = ([\d\.]+), Accuracy Combined = ([\d\.]+), Accuracy Action Name = ([\d\.]+), Accuracy Arguments = ([\d\.]+)\]"
    )

    # Lists to store extracted data
    epochs = []
    train_loss_action_name, train_loss_arguments = [], []
    train_acc_action_name, train_acc_arguments = [], []
    test_loss_action_name, test_loss_arguments = [], []
    test_acc_action_name, test_acc_arguments = [], []

    # Extract data from log file
    for line in log_data:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs.append(current_epoch)

        train_match = train_pattern.search(line)
        if train_match:
            train_loss_action_name.append(float(train_match.group(2)))
            train_loss_arguments.append(float(train_match.group(3)))
            train_acc_action_name.append(float(train_match.group(5)))
            train_acc_arguments.append(float(train_match.group(6)))

        test_match = test_pattern.search(line)
        if test_match:
            test_loss_action_name.append(float(test_match.group(2)))
            test_loss_arguments.append(float(test_match.group(3)))
            test_acc_action_name.append(float(test_match.group(5)))
            test_acc_arguments.append(float(test_match.group(6)))

    # Ensure all lists have the same length
    min_length = min(
        len(epochs),
        len(train_loss_action_name),
        len(train_loss_arguments),
        len(train_acc_action_name),
        len(train_acc_arguments),
        len(test_loss_action_name),
        len(test_loss_arguments),
        len(test_acc_action_name),
        len(test_acc_arguments),
    )
    epochs = epochs[:min_length]
    train_loss_action_name = train_loss_action_name[:min_length]
    train_loss_arguments = train_loss_arguments[:min_length]
    train_acc_action_name = train_acc_action_name[:min_length]
    train_acc_arguments = train_acc_arguments[:min_length]
    test_loss_action_name = test_loss_action_name[:min_length]
    test_loss_arguments = test_loss_arguments[:min_length]
    test_acc_action_name = test_acc_action_name[:min_length]
    test_acc_arguments = test_acc_arguments[:min_length]

    # Create DataFrame
    df = pd.DataFrame({
        "Epoch": epochs,
        "Train Loss Action Name": train_loss_action_name,
        "Train Loss Arguments": train_loss_arguments,
        "Train Accuracy Action Name": train_acc_action_name,
        "Train Accuracy Arguments": train_acc_arguments,
        "Test Loss Action Name": test_loss_action_name,
        "Test Loss Arguments": test_loss_arguments,
        "Test Accuracy Action Name": test_acc_action_name,
        "Test Accuracy Arguments": test_acc_arguments,
    })

    # Create a figure with two subplots for Loss
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Loss for Action Name
    axes[0].plot(df["Epoch"], df["Train Loss Action Name"], label="Train Loss Action Name", marker="o")
    axes[0].plot(df["Epoch"], df["Test Loss Action Name"], label="Test Loss Action Name", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss for Action Name over Epochs")
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer values on x-axis
    axes[0].legend()

    # Plot Loss for Arguments
    axes[1].plot(df["Epoch"], df["Train Loss Arguments"], label="Train Loss Arguments", marker="o")
    axes[1].plot(df["Epoch"], df["Test Loss Arguments"], label="Test Loss Arguments", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss for Arguments over Epochs")
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer values on x-axis
    axes[1].legend()

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_BASE + "loss_combined.png")
    print(f"Saved combined loss plot to '{OUTPUT_FILE_BASE}loss_combined.png'.")

    # Create a figure with two subplots for Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Accuracy for Action Name
    axes[0].plot(df["Epoch"], df["Train Accuracy Action Name"], label="Train Accuracy Action Name", marker="o")
    axes[0].plot(df["Epoch"], df["Test Accuracy Action Name"], label="Test Accuracy Action Name", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy for Action Name over Epochs")
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer values on x-axis
    axes[0].legend()

    # Plot Accuracy for Arguments
    axes[1].plot(df["Epoch"], df["Train Accuracy Arguments"], label="Train Accuracy Arguments", marker="o")
    axes[1].plot(df["Epoch"], df["Test Accuracy Arguments"], label="Test Accuracy Arguments", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy for Arguments over Epochs")
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Ensure integer values on x-axis
    axes[1].legend()

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE_BASE + "accuracy_combined.png")
    print(f"Saved combined accuracy plot to '{OUTPUT_FILE_BASE}accuracy_combined.png'.")


def run_loss_viz():
    try:
        plot_args_seperately()
    except:
        plot_args_combined()

if __name__ == "__main__":
    run_loss_viz()