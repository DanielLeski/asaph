## Importing data
To create an Asaph project, we first need to import the data.  A minimal command for importing biallelic SNPs from a VCF file would look like so:

    bin/import --vcf <path/to/vcf> \
               --populations <path/to/populations_file> \
               --workdir <path/to/workdir>

Asaph currently supports encoding SNPs as features in two ways: categories and counts.  In the categorical scheme, each genotype (e.g., A/T, A/A, T/T) is represented as a binary variable. In the counts scheme, each allele (e.g., A, T) is represented as an integer giving the number of copies (e.g., 0, 1, 2) of each allele the sample has.  The default and recommended scheme is to use categories.

A file listing the sample ids for each population must also be specified.  The populations file contains one group per line, with the first column indicating the population name, and the sample names separated by commas like so:

    Population 1,Sample1,Sample2
    Population 2,Sample3,Sample4

The sample ids have to match the ids in the VCF file.

The work directory will be created and contain the resulting Asaph data structures such as a feature matrix, feature labels, and sample labels.

To reduce disk space, Asaph can write the feature matrix as a zip file.  Compression can be enabled with the `--compress` flag:

    bin/import --vcf <path/to/vcf> \
               --populations <path/to/populations_file> \
               --workdir <path/to/workdir> \
               --compress


## SNP Rankings with Random Forests Variable Importance Scores
Asaph's original purpose, which it has since outgrown, was to support calculation of variable importances scores and ranking of SNPs using Random Forests.  Once data is imported, Random Forest models can be trained with the command:

    bin/random_forests --workdir <path/to/workdir> \
                       train \
                       --trees <number of trees> \
                       --populations <populations file>


Generally, you will want to sweep over the trees parameter, so you'll run the above command with a range of values for the `trees` parameter.  Asaph actually trains two Random Forests each time, to use in checking convergence.  You can check the convergence of the SNP rankings using the `analyzing-rankings` mode:

    bin/random_forests --workdir <path/to/workdir> \
                       analyze-rankings
                       

The `analyze-rankings` mode will generate two plots, comparisons of the number of SNPs used and the agreement of the top 0.01%, 0.1%, 1%, and 10% of ranked SNPs between each pair of models.  The plots are written in PDF and PNG formats and stored in `<workdir>/figures`. If the rankings do not demonstrate convergence, run the training command with a larger number of trees.  Once the rankings have converged, you can output the rankings to a text file:

    bin/random_forests --workdir <path/to/workdir> \
                       output-rankings \
                       --trees <select model with this many trees>

The rankings will be output to a text file in the `<workdir>/rankings` directory.



## SNP Ranking From Logistic Regression Weights (Ridge and Lasso)
Asaph can also be used for training ensembles of Logistic Regression models.  By training an ensemble and averaging over the feature weights, we can ensure that the rankings of the SNPs are consistent.  The LR workflow follows the RF workflow.  Once data is imported, you can train a LR model like so:

    bin/logistic_regression --workdir <path/to/workdir> \
                            train \
                            --populations <populations file>
                            --n-models <number of models>

Convergence of the SNP rankings can be evaluated like with the command:

    bin/logistic_regression --workdir <path/to/workdir> \
                            analyze-rankings

The `analyze-rankings` mode will generate two plots, comparisons of the number of SNPs used and the agreement of the top 0.01%, 0.1%, 1%, and 10% of ranked SNPs between each pair of models.  The plots are written in PDF and PNG formats and stored in `<workdir>/figures`. If the rankings do not demonstrate convergence, run the training command with a larger number of models.  Once the rankings have converged, you can output the rankings to a text file:

    bin/logistic_gression --workdir <path/to/workdir> \
                          output-rankings \
                          --n-models <select ensemble with this many models>

The rankings will be output to a text file in the `<workdir>/rankings` directory.

By default, LR models are trained with Stochastic Gradient Descent (SGD) and a L2 penalty.  Asaph also supports the average Stochastic Gradient Descent (ASGD) and Stochastic Average Gradient Descent (SAG) optimization algorithms. SGD additionally supports the elastic-net penalty. You can select different methods by using the `--methods` flag. If you do so, you'll need to use `--methods` each time you invoke `train`, `analyze-rankings`, and `output-rankings`.

You can also enable bagging, where the dataset is bootstrapped before each model is trained, with the `--bagging` flag for the `train` function. Bagging is disabled by default since we have found little impact from its usage.
