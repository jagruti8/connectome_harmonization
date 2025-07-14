""" Module with basic functions"""

import numpy as np
import random
import networkx as nx
import torch
import torch.nn.functional as F

def set_seed(seed):

    # Optionally set PYTHONHASHSEED to ensure consistent hash behavior
    #os.environ['PYTHONHASHSEED'] = str(seed)

    # Set seeds for Python random, NumPy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU

    # Ensure reproducibility with CUDA
    #torch.use_deterministic_algorithms(True)
    #torch.backends.cuda.matmul.allow_tf32 = False
    #torch.backends.cudnn.allow_tf32 = False
    #torch.backends.cudnn.deterministic = True  # Ensures that the same inputs yield the same outputs
    #torch.backends.cudnn.benchmark = False     # Disables some optimizations for reproducibility

    # Fill uninitialized memory with a known pattern for added determinism
    #torch.utils.deterministic.fill_uninitialized_memory=True

    # If you're on a GPU, synchronize to ensure order of operations
    #if torch.cuda.is_available():
    #    torch.cuda.synchronize()


def entropy_of_confusion_matrix(confusion_matrix):
    """
    Calculates the entropy of a confusion matrix.

    Args:
     confusion_matrix: A numpy array representing the confusion matrix.

    Returns:
     The entropy of the confusion matrix.
    """

    # Normalize the confusion matrix
    normalized_cm = confusion_matrix / confusion_matrix.sum()

    # Filter out zero probabilities to avoid log(0) issues
    normalized_cm = normalized_cm[normalized_cm > 0]

    # Calculate entropy
    entropy = -np.sum(normalized_cm * np.log2(normalized_cm))

    return entropy.item()
def entropy_of_confusion_matrix_row_wise(confusion_matrix):
    """
        Calculate the entropy of a confusion matrix.

        Parameters:
        confusion_matrix (np.array): A confusion matrix where each element (i, j)
                                     represents the number of times class i was predicted as class j.

        Returns:
        float: The calculated entropy for the confusion matrix.
        """
    row_entropies = []

    # Iterate through each row in the confusion matrix
    for row in confusion_matrix:
        # Normalize the row to get probabilities
        row_sum = np.sum(row)
        if row_sum == 0:
            row_entropies.append(0)
            continue  # skip rows with no data
        normalized_row = row / row_sum

        # Filter out zero probabilities to avoid log(0) issues
        normalized_row = normalized_row[normalized_row > 0]

        # Calculate entropy for this row
        entropy_row = -np.sum(normalized_row * np.log2(normalized_row))
        row_entropies.append(entropy_row)

    # Return the mean entropy across all rows
    return np.mean(row_entropies)

def check_unique_and_no_intersection(lists):
    # Step 1: Check if each list has unique elements
    for i, lst in enumerate(lists):
        if len(lst) != len(set(lst)):
            print(f"List {i + 1} does not have unique elements.")
            return False

    # Step 2: Check if there is no intersection between any pair of lists
    combined_elements = set()
    for i, lst in enumerate(lists):
        current_set = set(lst)
        if not combined_elements.isdisjoint(current_set):
            print(f"List {i + 1} has intersection with one of the previous lists.")
            return False
        combined_elements.update(current_set)

    # If all checks pass
    print("All lists have unique elements, and there is no intersection between any pair.")
    return True

# Custom loss function that applies different penalties based on the cost matrix
def custom_cross_entropy_loss(outputs, targets, cost_matrix):

    # Compute standard cross-entropy loss (logits version)
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')

    # Get the predicted and true class indices
    pred_classes = torch.argmax(outputs, dim=1)

    # Gather penalties from the cost matrix based on predictions and true labels
    penalties = cost_matrix[targets, pred_classes]

    # Apply the penalties to the cross-entropy loss
    weighted_loss = ce_loss * penalties

    # Return the mean loss
    return weighted_loss.mean()

def pearson_correlation(output, target):
    vx = output - torch.mean(output) + 1e-8
    vy = target - torch.mean(target) + 1e-8

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) )
    return torch.clamp(cost, min=-1.0, max=1.0)

def generate_random_mask(shape):
    """Generate a random binary mask with the given shape."""
    return np.random.randint(0, 2, shape)  # Random 0 or 1 for each element

def combine_matrices(adj_matrix1, adj_matrix2):
    """Combine two matrices using a random binary mask."""
    mask = generate_random_mask(adj_matrix1.shape)
    adj_combined_matrix = np.where(mask == 0, adj_matrix1, adj_matrix2)  # Take from matrix1 if mask is 0, else from matrix2
    return adj_combined_matrix

def combine_vectors(vector1, vector2):
    """Combine two vectors using a random binary mask."""
    mask = generate_random_mask(vector1.shape)
    combined_vector = np.where(mask == 0, vector1, vector2)  # Take from vector1 if mask is 0, else from vector2
    return combined_vector

def generate_random_mask_equal(shape):
    """Generate a random binary mask with the given shape where 50% elements are 0 and the rest 50% are 1."""

    # Total size of the mask
    size = np.prod(shape)  # Total number of elements in the mask

    # Create a mask with equal numbers of 0s and 1s
    mask = np.array([0] * (size // 2) + [1] * (size // 2))

    # Shuffle the mask to distribute 0s and 1s randomly
    np.random.shuffle(mask)

    # Reshape the mask
    mask = mask.reshape(shape)

    return mask

def generate_almost_random_mask_equal(shape):
    """Generate a random binary mask with the given shape where almost 50% elements are 0 and the rest are 1."""

    # Total size of the mask
    size = np.prod(shape)  # Total number of elements in the mask

    # Create a mask with almost equal numbers of 0s and 1s
    mask = np.array([0] * int(np.ceil(size/2)) + [1] * int(np.floor(size/2)))

    # Shuffle the mask to distribute 0s and 1s randomly
    np.random.shuffle(mask)

    # Reshape the mask
    mask = mask.reshape(shape)

    return mask

def combine_matrices_equal(adj_matrix1, adj_matrix2):
    """Combine two matrices with equal contributions using a random binary mask."""
    mask = generate_random_mask_equal(adj_matrix1.shape)
    adj_combined_matrix = np.where(mask == 0, adj_matrix1, adj_matrix2)  # Take from matrix1 if mask is 0, else from matrix2
    return adj_combined_matrix

def combine_vectors_equal(vector1, vector2):
    """Combine two vectors with almost equal contributions using a random binary mask."""
    mask = generate_almost_random_mask_equal(vector1.shape)
    combined_vector = np.where(mask == 0, vector1, vector2)  # Take from vector1 if mask is 0, else from vector2
    return combined_vector

def combine_matrices_linear(adj_matrix1, adj_matrix2, lambda1=0.5):
    """Combine two matrices in a linear interpolation way."""
    adj_combined_matrix = (1-lambda1)*adj_matrix1 + (lambda1)*adj_matrix2
    adj_combined_matrix = np.round(np.where(adj_combined_matrix<0, 0 , adj_combined_matrix))
    return adj_combined_matrix

def combine_vectors_linear(vector1, vector2, lambda1=0.5):
    """Combine two vectors in a linear interpolation way."""
    combined_vector = (1-lambda1)*vector1 + (lambda1)*vector2
    combined_vector = np.round(np.where(combined_vector<0, 0 , combined_vector))
    return combined_vector


def find_matrix_shape_from_upper_triangular(upper_triangular_tensor):
    """
    Given the upper triangular part of a matrix as a 1D tensor, compute the size of the original square matrix.

    Args:
        upper_triangular_tensor (torch.Tensor): A 1D tensor containing the upper triangular elements.

    Returns:
        int: The size (n x n) of the original square matrix.
    """
    # Number of elements in the upper triangular tensor
    k = upper_triangular_tensor.size(0)

    # Solve for n using the quadratic formula: n = (1 + sqrt(1 + 8k)) / 2
    n = int((1 + np.sqrt(1 + 8 * k)) / 2)

    return n

def find_matrix_shape_from_upper_triangular_numpy(upper_triangular_vector):
    """
    Given the upper triangular part of a matrix as a 1D vector, compute the size of the original square matrix.

    Args:
        upper_triangular_vector (numpy): A 1D vector containing the upper triangular elements.

    Returns:
        int: The size (n x n) of the original square matrix.
    """
    # Number of elements in the upper triangular vector
    k = upper_triangular_vector.shape[0]

    # Solve for n using the quadratic formula: n = (1 + sqrt(1 + 8k)) / 2
    n = int((1 + np.sqrt(1 + 8 * k)) / 2)

    return n

def matrix_to_graph(adj_matrix):
    """Convert an adjacency matrix to a NetworkX graph."""
    graph = nx.from_numpy_array(adj_matrix)
    return graph

def nodal_strength_loss(target_matrices, predicted_matrices):
    # Calculate nodal strength (weighted degree) for each node
    target_strengths = calculate_nodal_strength(target_matrices)
    predicted_strengths = calculate_nodal_strength(predicted_matrices)

    # L1 loss between target and predicted nodal strengths
    loss = torch.mean(torch.abs(target_strengths - predicted_strengths))
    return loss

def calculate_nodal_strength(adj_matrices):
    """
    Calculate nodal strength (sum of edge weights for each node).
    adj_matrices: torch.Tensor (B, N, N), adjacency matrix of B graphs.
    """
    return torch.sum(adj_matrices, dim=2)  # Sum of each row (or column for undirected graphs or symmetric matrices)

def kl_divergence_nodal_strength(target_matrices, predicted_matrices):
    """
    Calculate KL divergence between nodal strength distributions of target and predicted graphs.
    target_matrices: torch.Tensor (B, N, N), target adjacency matrices
    predicted_matrices: torch.Tensor (B, N, N), predicted adjacency matrices
    """
    epsilon = 1e-10  # Small value to prevent division by zero

    # Step 1: Calculate nodal strengths
    target_strengths = calculate_nodal_strength(target_matrices)
    predicted_strengths = calculate_nodal_strength(predicted_matrices)

    # Step 2: Convert strengths to probability distributions
    target_probs = target_strengths / (target_strengths.sum(dim=1, keepdim=True) + epsilon)
    predicted_probs = predicted_strengths / (predicted_strengths.sum(dim=1, keepdim=True) + epsilon)

    # Step 3: Convert pred_probs to log-probabilities (required for KL divergence in PyTorch)
    predicted_log_probs = torch.log(predicted_probs + epsilon)  # Adding epsilon for numerical stability

    # Step 4: Calculate KL divergence (target is expected in probability form, pred in log-probability form)
    kl_div = F.kl_div(predicted_log_probs, target_probs, reduction='batchmean')

    return kl_div

def compute_laplacian(adj_matrix):
    """
    Compute the Laplacian matrix for a given adjacency matrix.

    Args:
    adj_matrix (torch.Tensor): Adjacency matrix of shape (num_nodes, num_nodes).

    Returns:
    torch.Tensor: Laplacian matrix of shape (num_nodes, num_nodes).
    """
    # Degree matrix (sum of adjacency matrix rows)
    degree = torch.diag(adj_matrix.sum(dim=-1))
    # Laplacian matrix
    laplacian = degree - adj_matrix
    return laplacian

def eigenvalue_difference_batch(target_matrices, predicted_matrices): #TODO - Later - sort the eigenvalues and then compare
    """
    Regularization loss based on the absolute difference between eigenvalues.

    Args:
    target_matrices (torch.Tensor): Target adjacency matrices (batch_size, num_nodes, num_nodes).
    predicted_matrices (torch.Tensor): Predicted adjacency matrices (batch_size, num_nodes, num_nodes).

    Returns:
    torch.Tensor: Regularization loss (absolute difference of eigenvalues).
    """
    # Compute Laplacians for target and predicted adjacency matrices
    target_laplacians = torch.stack([compute_laplacian(adj_matrix) for adj_matrix in target_matrices])
    predicted_laplacians = torch.stack([compute_laplacian(adj_matrix) for adj_matrix in predicted_matrices])

    # Compute eigenvalues for all adjacency matrices in the batch
    target_eigenvalues = torch.linalg.eigvalsh(target_laplacians)  # Shape: (batch_size, num_nodes)
    predicted_eigenvalues = torch.linalg.eigvalsh(predicted_laplacians)  # Shape: (batch_size, num_nodes)

    # Compute the absolute difference and average over the batch
    regularization_loss = torch.mean(torch.abs(target_eigenvalues - predicted_eigenvalues))

    return regularization_loss

def laplacian_matrix(A):
    """Computes the standard Laplacian: L = D - A"""
    D = np.diag(np.sum(A, axis=1))
    return D - A

def normalized_laplacian(A):
    """Computes the normalized Laplacian: L_norm = I - D^(-1) A"""
    D = np.diag(np.sum(A, axis=1))
    D_inv = np.linalg.inv(D)  # Inverse of D
    return np.eye(A.shape[0]) - D_inv @ A

def symmetric_normalized_laplacian(A):
    """Computes the symmetric normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)"""
    D = np.diag(np.sum(A, axis=1))
    D_sqrt_inv = np.linalg.inv(np.sqrt(D))  # D^(-1/2)
    return np.eye(A.shape[0]) - D_sqrt_inv @ A @ D_sqrt_inv

def compute_eigenvalue(matrix, laplacian_arg='standard'):
    """
    Computes the eigenvalue of a matrix
    :param matrix:
    :param laplacian_arg:
    :return:
    """
    if laplacian_arg == "standard":
        L_matrix = laplacian_matrix(matrix)
    elif laplacian_arg == "norm":
        L_matrix = normalized_laplacian(matrix)
    elif laplacian_arg == "sym":
        L_matrix = symmetric_normalized_laplacian(matrix)
    else:
        L_matrix = matrix

    eig_matrix = np.linalg.eigvals(L_matrix)

    return eig_matrix

def eigenvalue_difference_laplacian(eig_target, eig_predicted):
    """
    Computes the mean absolute difference and mean square difference between the eigenvalues of the (Laplacians) of the target and predicted matrices.

    Args:
    - eig_target (numpy.ndarray): (Laplacian) Ground truth adjacency matrix eigen values
    - eig_predicted (numpy.ndarray): (Laplacian) Predicted adjacency matrix eigen values

    Returns:
        tuple: A tuple containing two real numbers.
            - eig_diff_l1 (float): Mean absolute differences between corresponding eigenvalues.
            - eig_diff_l2 (float): Mean squared differences between corresponding eigenvalues.
    """

    # Sort eigenvalues to ensure correct pairing
    eig_target = np.sort(eig_target)
    eig_predicted = np.sort(eig_predicted)

    # Compute mean absolute eigenvalue difference and the mean square eigenvalue difference
    eig_diff_l1 = np.mean(np.abs(eig_target - eig_predicted))
    eig_diff_l2 = np.mean(np.square(eig_target - eig_predicted))

    return eig_diff_l1, eig_diff_l2

def eigenvalue_difference_laplacian_batch(eig_targets, eig_predicteds):
    """
    Computes the mean absolute difference and mean square difference
    between the eigenvalues of the Laplacians of multiple target and predicted matrices.

    Args:
    - eig_targets (list of numpy.ndarray): List of Laplacian eigenvalues for ground truth matrices.
    - eig_predicteds (list of numpy.ndarray): List of Laplacian eigenvalues for predicted matrices.

    Returns:
        tuple: Two lists containing:
            - eig_diff_l1 (list of float): Mean absolute differences for each matrix pair.
            - eig_diff_l2 (list of float): Mean squared differences for each matrix pair.
    """
    eig_diff_l1 = []
    eig_diff_l2 = []

    for eig_target, eig_predicted in zip(eig_targets, eig_predicteds):
        # Sort eigenvalues to ensure correct pairing
        eig_target = np.sort(eig_target)
        eig_predicted = np.sort(eig_predicted)

        # Compute mean absolute and mean squared differences
        eig_diff_l1.append((np.abs(eig_target - eig_predicted)))
        eig_diff_l2.append((np.square(eig_target - eig_predicted)))

    eig_diff_l1 = np.mean(np.stack(eig_diff_l1))
    eig_diff_l2 = np.mean(np.stack(eig_diff_l2))

    return eig_diff_l1, eig_diff_l2

