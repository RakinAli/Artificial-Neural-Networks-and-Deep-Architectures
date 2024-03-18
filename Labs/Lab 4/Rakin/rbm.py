from util import *
from tqdm import tqdm
import os

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size

        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        

        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))

        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        

        self.weight_v_to_h = None

        self.weight_h_to_v = None

        self.learning_rate = 0.01

        self.momentum = 0.7

        self.print_period = self.batch_size * 100

        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }

        return

    def cd1(self, visible_trainset, n_iterations=100, plotting=False):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
          plotting: set to True to plot (default = True)
        """

        n_samples = visible_trainset.shape[0]
        loss_list = []
        results_list = []
        error = []  # Storing error per iteration
        elements = int(n_samples / self.batch_size)
        current_epoch = 1  # Initialize current epoch as the first one
        for epoch in range(n_iterations):
            for it in tqdm(range(elements)):

                # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.

                index_init = int(it % elements)
                index_stop = int((index_init + 1) * self.batch_size)
                index_init *= self.batch_size
                v_0 = visible_trainset[index_init:index_stop, :]
                # Positive phase v_0 -> h_0
                p_h_given_v_0, h_0 = self.get_h_given_v(v_0)

                # Negative phase h_0 -> v_1 -> h_1
                p_v_given_h_1, v_1 = self.get_v_given_h(h_0)
                p_h_given_v_0, h_1 = self.get_h_given_v(v_1)

                # [TODO TASK 4.1] update the parameters using function 'update_params'

                self.update_params(v_0, h_0, v_1, h_1)

                if plotting:
                    # Plot for each batch
                    if it % self.batch_size == 0:
                        hidden_restored = self.get_h_given_v(v_0)[1]
                        restored_image = self.get_v_given_h(hidden_restored)[1]
                        error.append(np.linalg.norm(v_0 - restored_image))

                    # Plot average loss for each epoch when epochs >= 10 and return it
                    if current_epoch >= 10:
                        # Restoring image
                        hidden_restored = self.get_h_given_v(v_0)[1]
                        restored_image = self.get_v_given_h(hidden_restored)[1]
                        loss_function = np.linalg.norm(v_0 - restored_image) / self.batch_size
                        loss_list.append(loss_function)  # Store last 10 percent
                        if it == elements - 1:
                            # Last iteration before emptying the list
                            results_list.append(np.array(loss_list).sum() / len(loss_list))  # Append avg loss epoch
                            loss_list = []  # Empty list

                if self.is_bottom:
                    # Visualize it on the last iteration of the last epoch
                    if epoch == n_iterations - 1 and it == elements - 1:
                        iterations = n_iterations * elements # Total iterations
                        viz_rf(
                            weights=self.weight_vh[:, self.rf["ids"]].reshape(
                                (self.image_size[0], self.image_size[1], -1)
                            ),
                            it=iterations,
                            grid=self.rf["grid"],
                        )

            current_epoch += 1  # Update current epoch

        if plotting:
            plt.plot(range(len(error)), error)
            plt.xlabel("Batch")
            plt.ylabel("Error")

            # Get current axes and limits
            ax = plt.gca()
            y_lim = ax.get_ylim()
            x_lim = ax.get_xlim()

            # Calculate position for the text
            text_x = x_lim[1] * 0.7  # 70% across the x-axis
            text_y = y_lim[1] * 0.9  # 90% up the y-axis

            # Place the text on the plot
            plt.text(text_x, text_y, f"Hidden layers: {self.ndim_hidden}, Batch size: {self.batch_size}, Iterations: {n_iterations}",
                    horizontalalignment='right', verticalalignment='top')

            # Save the file as Error_hidden_layers_batch_size_iterations.png
            figure_name = f"Hidden_layers_{self.ndim_hidden}_batch_{self.batch_size}_iterations_{n_iterations}.png"
            figure_path = "plots"
            # Ensure the 'plots' directory exists
            if not os.path.exists(figure_path):
                os.makedirs(figure_path)

            # Save the figure in the 'plots' directory
            plt.savefig(os.path.join(figure_path, figure_name))

        return results_list


    def update_params(self,v_0,h_0,v_k,h_k):
        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO-Done TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters

        # The gradients calculations
        self.delta_bias_v = self.learning_rate * (np.sum(v_0 - v_k, axis=0)) # / self.batch_size)

        self.delta_weight_vh = self.learning_rate * ((v_0.T @ h_0)-(v_k.T @ h_k)) # / self.batch_size)

        self.delta_bias_h = self.learning_rate * (np.sum(h_0 - h_k, axis=0)) # / self.batch_size)

        # Actually updating the parameters
        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        return

    def get_h_given_v(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_vh is not None
        n_samples = visible_minibatch.shape[0]

        # [TODO-Done TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below)
        p_h_given_v = sigmoid(visible_minibatch @ self.weight_vh + self.bias_h)
        h = sample_binary(p_h_given_v)

        return p_h_given_v, h

    def get_v_given_h(self,hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO-Done TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            total_input = hidden_minibatch @ self.weight_vh.T + self.bias_v

            # Create arrays
            p_v_given_h = np.ndarray(shape=total_input.shape)
            s = np.ndarray(shape=total_input.shape)

            # Compute probabilities and activations for visible layer
            p_v_given_h[:,:-self.n_labels] = sigmoid(total_input[:,:-self.n_labels])
            p_v_given_h[:, -self.n_labels:] = softmax(total_input[:, -self.n_labels:])

            s[:,:-self.n_labels] = sample_binary(p_v_given_h[:,:-self.n_labels])
            s[:, -self.n_labels:] = sample_categorical(p_v_given_h[:, -self.n_labels:])

        else:

            # [TODO-Done TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)
            p_v_given_h = sigmoid(hidden_minibatch @ self.weight_vh.T + self.bias_v)
            s = sample_binary(p_v_given_h)

        return p_v_given_h, s

    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):

        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self, visible_minibatch):
        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"

        Args:
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:
           tuple ( p(h|v) , h)
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # finished
        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (
        #  replace the zeros below)
        p_h_given_v_dir = sigmoid(
            np.dot(visible_minibatch, self.weight_v_to_h) + self.bias_h
        )
        h = sample_binary(p_h_given_v_dir)

        return p_h_given_v_dir, h

    def get_v_given_h_dir(self,hidden_minibatch):
        # Maths notation in overleaf:
        # p(h_j = 1 | \mathbf{v}) = \sigma\left(\sum_{i} v_i W_{ij}^{(v \rightarrow h)} + b_j^h\right)

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] - DONE
            # Appropriate code here is to raise an error (replace pass below)
            print("Error: This function should not be executed")
            raise ValueError("This function should not be executed")

        else:

            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)

            p_v_given_h_dir = sigmoid(hidden_minibatch @ self.weight_h_to_v.T + self.bias_v)
            sample = sample_binary(p_v_given_h_dir)

        return p_v_given_h_dir, sample

    def update_generate_params(self,inps,trgs,preds):

        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0

        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 

        return

    def update_recognize_params(self,inps,trgs,preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h

        return    
