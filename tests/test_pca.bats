#!/usr/bin/env bats

load model_setup_helper

@test "Run pca with no arguments" {
    run ${BATS_TEST_DIRNAME}/../bin/pca
    [ "$status" -eq 2 ]
}

@test "Run pca with --help option" {
    run ${BATS_TEST_DIRNAME}/../bin/pca --help
    [ "$status" -eq 0 ]
}

@test "pca (categories)" {
    run ${BATS_TEST_DIRNAME}/../bin/pca \
        --workdir ${WORKDIR_PATH} \
        train \
        --n-components 6

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/models/pca.pkl" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        explained-variance-analysis

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/figures/pca_explained_variance_ratios.png" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        plot-projections \
        --pairs 1 2 3 4

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/figures/pca_projection_1_2.png" ]
    [ -e "${WORKDIR_PATH}/figures/pca_projection_3_4.png" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        output-coordinates \
        --selected-components 1 2 3 4 \
        --output-fl ${WORKDIR_PATH}/pca_coordinates.txt

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/pca_coordinates.txt" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        pop-association-tests

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/population_pca_association_tests.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        output-loading-magnitudes \
        --components 1 2

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/pca_loading_magnitudes.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        output-loading-factors \
        --components 1 2

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/pca_loading_factors.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        snp-association-tests \
        --components 1 2 \
        --model-type logistic

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/snp_pc_1_logreg_assoc_tests.tsv" ]
    [ -e "${WORKDIR_PATH}/analysis/snp_pc_2_logreg_assoc_tests.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        snp-association-tests \
        --components 1 2 \
        --model-type linear

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/snp_pc_1_linreg_assoc_tests.tsv" ]
    [ -e "${WORKDIR_PATH}/analysis/snp_pc_2_linreg_assoc_tests.tsv" ]
    
    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        sweep-clusters \
        --components 1 2 \
        --n-clusters 2 4 6 8

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/figures/cluster_inertia_1_2.png" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        cluster-samples \
        --components 1 2 \
        --n-clusters 2

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/clusters_2.tsv" ]
}

@test "pca (counts)" {
    run ${BATS_TEST_DIRNAME}/../bin/pca \
        --workdir ${COUNTS_WORKDIR_PATH} \
        train \
        --n-components 6

    [ "$status" -eq 0 ]
    [ -e "${COUNTS_WORKDIR_PATH}/models/pca.pkl" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${COUNTS_WORKDIR_PATH} \
        explained-variance-analysis

    [ "$status" -eq 0 ]
    [ -e "${COUNTS_WORKDIR_PATH}/figures/pca_explained_variance_ratios.png" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${COUNTS_WORKDIR_PATH} \
        plot-projections \
        --pairs 1 2 3 4

    [ "$status" -eq 0 ]
    [ -e "${COUNTS_WORKDIR_PATH}/figures/pca_projection_1_2.png" ]
    [ -e "${COUNTS_WORKDIR_PATH}/figures/pca_projection_3_4.png" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${COUNTS_WORKDIR_PATH} \
        output-coordinates \
        --selected-components 1 2 3 4 \
        --output-fl ${COUNTS_WORKDIR_PATH}/pca_coordinates.txt

    [ "$status" -eq 0 ]
    [ -e "${COUNTS_WORKDIR_PATH}/pca_coordinates.txt" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${COUNTS_WORKDIR_PATH} \
        pop-association-tests

    [ "$status" -eq 0 ]
    [ -e "${COUNTS_WORKDIR_PATH}/analysis/population_pca_association_tests.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${COUNTS_WORKDIR_PATH} \
        sweep-clusters \
        --components 1 2 \
        --n-clusters 2 4 6 8
    
    [ "$status" -eq 0 ]
    [ -e "${COUNTS_WORKDIR_PATH}/figures/cluster_inertia_1_2.png" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${COUNTS_WORKDIR_PATH} \
        cluster-samples \
        --components 0 1 \
        --n-clusters 2
    
    [ "$status" -eq 0 ]
    [ -e "${COUNTS_WORKDIR_PATH}/analysis/clusters_2.tsv" ]
}

@test "nmf (categories)" {
    run ${BATS_TEST_DIRNAME}/../bin/pca \
        --workdir ${WORKDIR_PATH} \
        train-nmf \
        --n-components 6 \
        --beta-loss frobenius \
        --solver cd

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/models/pca.pkl" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        plot-projections \
        --pairs 1 2 3 4

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/figures/pca_projection_1_2.png" ]
    [ -e "${WORKDIR_PATH}/figures/pca_projection_3_4.png" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        output-coordinates \
        --selected-components 1 2 3 4 \
        --output-fl ${WORKDIR_PATH}/pca_coordinates.txt

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/pca_coordinates.txt" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        pop-association-tests

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/population_pca_association_tests.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        output-loading-magnitudes \
        --components 1 2

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/pca_loading_magnitudes.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        output-loading-factors \
        --components 1 2

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/pca_loading_factors.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        snp-association-tests \
        --components 1 2 \
        --model-type logistic

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/snp_pc_1_logreg_assoc_tests.tsv" ]
    [ -e "${WORKDIR_PATH}/analysis/snp_pc_2_logreg_assoc_tests.tsv" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        snp-association-tests \
        --components 1 2 \
        --model-type linear

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/snp_pc_1_linreg_assoc_tests.tsv" ]
    [ -e "${WORKDIR_PATH}/analysis/snp_pc_2_linreg_assoc_tests.tsv" ]
    
    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        sweep-clusters \
        --components 1 2 \
        --n-clusters 2 4 6 8

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/figures/cluster_inertia_1_2.png" ]

    run ${BATS_TEST_DIRNAME}/../bin/pca \
	    --workdir ${WORKDIR_PATH} \
        cluster-samples \
        --components 1 2 \
        --n-clusters 2

    [ "$status" -eq 0 ]
    [ -e "${WORKDIR_PATH}/analysis/clusters_2.tsv" ]
}

