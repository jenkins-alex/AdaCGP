import matplotlib.pyplot as plt
import numpy as np
import os

def save_figures(results, weight_matrix, save_path):

    # plot the error curves
    fig = plt.figure(figsize=(10, 4))
    plt.plot(results['pred_error'], label='Prediction Error (from Psi)')
    plt.plot(results['pred_error_recursive_moving_average'][1:], label='Prediction Error Moving Average (from Psi)')
    plt.plot(results['filter_error'], label='Graph Filter Error')
    plt.plot(results['w_error'], label='W Error')
    plt.plot(results['percentage_correct_elements'], label='ptg. correct elements')
    plt.plot(results['p_miss'], label='p_miss')
    plt.plot(results['p_false_alarm'], label='p_false_alarm')

    # plot the convergence status (True, False) as block blue
    alg1_converged_status = np.array(results['first_alg_converged_status'])
    alg1_converged_status = alg1_converged_status * 1
    plt.plot(alg1_converged_status * 1.5, label='Converged alg1')

    alg2_converged_status = np.array(results['second_alg_converged_status'])
    alg2_converged_status = alg2_converged_status * 1
    plt.plot(alg2_converged_status * 1.5, label='Converged alg2')

    # plot the losses
    plt.ylim(0, 1.51)
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'error_curves.png'))
    plt.close()

    # plot weight matrix
    W = results['matrices'][-1]
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(weight_matrix.detach().cpu().numpy())
    axs[1].imshow(W)
    axs[2].imshow(W != 0, cmap='binary_r')
    axs[0].set_title(f'True W')
    axs[1].set_title(f'Pred W')
    axs[2].set_title(f'Pred W != 0')
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(os.path.join(save_path, 'weight_matrix.png'))
    plt.close()
    print("Figures saved.")
