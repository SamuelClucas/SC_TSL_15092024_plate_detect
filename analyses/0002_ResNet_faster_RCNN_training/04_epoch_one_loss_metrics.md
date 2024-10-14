# Diagnostic: Inspecting One Epoch Evaluation Loss Metrics

2/10/25

**Following on from analysis [03](03_ResNet50_setup.qmd):**  
After one epoch of training, I now have some insight into how the code
is behaving. Firstly, [standard error](results/03_error.txt) from the
final code block of analysis 03 indicates a problem with the
‘plot_eval_metrics’ function from analysis
[02](02_helper_training_functions.md). I think this is relatively benign
and simple to fix. It’s a new function I wrote hastily and I will
rewrite it.  

More importantly though, I first want to inspect the loss_classifier,
loss_objectness, loss_rpn_box_reg and loss metrics produced by the call
to ‘evaluate()’ again defined in analysis
[02](02_helper_training_functions.md). This function uses utility
modules found in the [torchvision_deps](../../src/torchvision_deps/)
folder. I think the data might hint at some unwanted behaviour in some
of the network modules which must be identified and corrected in future
development.  

### Purpose:

To create plots from one epoch of training from analysis
[03](02_helper_training_functions.md)’s standard output and explore the
implications of these data on the functionality of analysis
[02](02_helper_training_functions.md)’s helper training functions, as
well as the utility modules in
[torchvision_deps](../../src/torchvision_deps/). From this, I must
correct any dysfunction prior to training on the cluster.

#### Redirecting loss metrics of interest from [03_std_out.txt](results/03_std_out.txt)…

Firstly, I used:  

``` {bash}
grep -o '.../445\|loss: .......\|loss_objectness: ......\|loss_rpn_box_reg: .......\|loss_classifier: results/stdout_ResNet_Training.txt > results/cleaned_output.txt
```

#### Creating plots from [this tidier output](results/cleaned_output.txt)…

``` python
import matplotlib.pyplot as plt

# Initialize lists to store the metrics
progression = []
loss = []
loss_objectness = []
loss_rpn_box_reg = []
loss_classifier = []

# Read the cleaned output file
with open('results/cleaned_output.txt', 'r') as file:
    for line in file:
        if '/445' in line:
            progression.append(int(line.split('/')[0]))
        elif 'loss: ' in line:
            loss.append(float(line.split('loss: ')[1]))
        elif 'loss_objectness: ' in line:
            loss_objectness.append(float(line.split('loss_objectness: ')[1]))
        elif 'loss_rpn_box_reg: ' in line:
            loss_rpn_box_reg.append(float(line.split('loss_rpn_box_reg: ')[1]))
        elif 'loss_classifier: ' in line:
            loss_classifier.append(float(line.split('loss_classifier: ')[1]))

# Plotting the metrics
plt.figure(figsize=(10, 6))
plt.plot(progression, loss, label='Loss')
plt.plot(progression, loss_objectness, label='Loss Objectness')
plt.plot(progression, loss_rpn_box_reg, label='Loss RPN Box Reg')
plt.plot(progression, loss_classifier, label='Loss Classifier')

plt.xlabel('Progression through Epoch')
plt.ylabel('Loss')
plt.yscale("log")
plt.title('Loss Metrics Progression Through One Epoch')
plt.legend()
plt.grid(True)
plt.show()
```

![](04_epoch_one_loss_metrics_files/figure-commonmark/cell-2-output-1.png)