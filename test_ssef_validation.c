#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "rknn_api.h"

#define NUM_SAMPLES 4624
#define NUM_FEATURES 131
#define NUM_CLASSES 38

#define INPUT_SCALE 0.019937178f
#define INPUT_ZERO_POINT (-13)

#define MODEL_PATH "/userdata/landmark_mlp.rknn"
#define FEATURES_PATH "/userdata/validation_features.bin"
#define LABELS_PATH "/userdata/validation_labels.bin"
#define CLASS_NAMES_PATH "/userdata/class_names.txt"

static const char *default_class_names[NUM_CLASSES] = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "L", "M", "O", "P", "Q", "R", "S", "T", "U", "V",
    "Y", "Z", "K", "N", "W", "X", "0", "HELLO", "BYE"
};

static void fail(const char *message)
{
    fprintf(stderr, "ERROR: %s\n", message);
    exit(EXIT_FAILURE);
}

static void check_rknn(int result, const char *operation)
{
    if (result != RKNN_SUCC) {
        fprintf(stderr, "ERROR: %s failed with code %d\n",
                operation, result);
        exit(EXIT_FAILURE);
    }
}

static void *read_file(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if (fseek(file, 0, SEEK_END) != 0)
        fail("fseek failed");

    long length = ftell(file);
    if (length < 0)
        fail("ftell failed");

    rewind(file);

    void *buffer = malloc((size_t)length);
    if (!buffer)
        fail("File allocation failed");

    if (fread(buffer, 1, (size_t)length, file) != (size_t)length)
        fail("File read failed");

    fclose(file);
    *size = (size_t)length;
    return buffer;
}

static int8_t quantize_input(float value)
{
    int q = (int)roundf(value / INPUT_SCALE) + INPUT_ZERO_POINT;

    if (q > 127) q = 127;
    if (q < -128) q = -128;

    return (int8_t)q;
}

int main(int argc, char **argv)
{
    const char *model_path = argc > 1 ? argv[1] : MODEL_PATH;
    const char *features_path = argc > 2 ? argv[2] : FEATURES_PATH;
    const char *labels_path = argc > 3 ? argv[3] : LABELS_PATH;

    printf("========================================\n");
    printf(" SSEF RKNN Validation Tester\n");
    printf("========================================\n");
    printf("Samples:  %d\n", NUM_SAMPLES);
    printf("Features: %d\n", NUM_FEATURES);
    printf("Classes:  %d\n\n", NUM_CLASSES);

    /* Load model. */
    size_t model_size = 0;
    void *model_data = read_file(model_path, &model_size);

    printf("Model: %s (%zu bytes)\n", model_path, model_size);

    rknn_context ctx = 0;
    check_rknn(
        rknn_init(&ctx, model_data, (uint32_t)model_size, 0, NULL),
        "rknn_init"
    );

    free(model_data);

    /* Query model input/output counts. */
    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));

    check_rknn(
        rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)),
        "query input/output count"
    );

    if (io_num.n_input != 1 || io_num.n_output != 1)
        fail("Expected exactly one input and one output tensor");

    /* Query tensor attributes. */
    rknn_tensor_attr input_attr;
    rknn_tensor_attr output_attr;

    memset(&input_attr, 0, sizeof(input_attr));
    memset(&output_attr, 0, sizeof(output_attr));

    input_attr.index = 0;
    output_attr.index = 0;

    check_rknn(
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR,
                   &input_attr, sizeof(input_attr)),
        "query input attributes"
    );

    check_rknn(
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR,
                   &output_attr, sizeof(output_attr)),
        "query output attributes"
    );

    printf("\nInput tensor:\n");
    printf("  Name: %s\n", input_attr.name);
    printf("  Dimensions: ");
    for (uint32_t i = 0; i < input_attr.n_dims; i++)
        printf("%u ", input_attr.dims[i]);
    printf("\n  Elements: %u\n", input_attr.n_elems);
    printf("  Size: %u\n", input_attr.size);
    printf("  Type: %s\n", get_type_string(input_attr.type));
    printf("  Scale: %.9f\n", input_attr.scale);
    printf("  Zero point: %d\n", input_attr.zp);

    printf("\nOutput tensor:\n");
    printf("  Name: %s\n", output_attr.name);
    printf("  Dimensions: ");
    for (uint32_t i = 0; i < output_attr.n_dims; i++)
        printf("%u ", output_attr.dims[i]);
    printf("\n  Elements: %u\n", output_attr.n_elems);
    printf("  Size: %u\n", output_attr.size);
    printf("  Type: %s\n", get_type_string(output_attr.type));
    printf("  Scale: %.9f\n", output_attr.scale);
    printf("  Zero point: %d\n\n", output_attr.zp);

    if (input_attr.n_elems != NUM_FEATURES)
        fail("Model input does not contain 131 features");

    if (output_attr.n_elems != NUM_CLASSES)
        fail("Model output does not contain 38 classes");

    if (input_attr.type != RKNN_TENSOR_INT8)
        fail("Expected INT8 input tensor");

    if (output_attr.type != RKNN_TENSOR_INT8)
        fail("Expected INT8 output tensor");

    if (fabsf(input_attr.scale - INPUT_SCALE) > 0.00001f ||
        input_attr.zp != INPUT_ZERO_POINT)
        fail("Input quantisation parameters do not match expected values");

    /* Load validation data. */
    size_t feature_bytes = 0;
    size_t label_bytes = 0;

    float *features = read_file(features_path, &feature_bytes);
    int32_t *labels = read_file(labels_path, &label_bytes);

    const size_t expected_feature_bytes =
        (size_t)NUM_SAMPLES * NUM_FEATURES * sizeof(float);

    const size_t expected_label_bytes =
        (size_t)NUM_SAMPLES * sizeof(int32_t);

    if (feature_bytes != expected_feature_bytes)
        fail("Validation feature file has incorrect size");

    if (label_bytes != expected_label_bytes)
        fail("Validation label file has incorrect size");

    printf("Validation files loaded successfully.\n");

    /* Allocate RKNN input/output memory. */
    uint32_t input_mem_size =
        input_attr.size_with_stride > input_attr.size
        ? input_attr.size_with_stride : input_attr.size;

    uint32_t output_mem_size =
        output_attr.size_with_stride > output_attr.size
        ? output_attr.size_with_stride : output_attr.size;

    rknn_tensor_mem *input_mem =
        rknn_create_mem(ctx, input_mem_size);

    rknn_tensor_mem *output_mem =
        rknn_create_mem(ctx, output_mem_size);

    if (!input_mem || !output_mem)
        fail("Failed to allocate RKNN tensor memory");

    /*
     * Input values are already quantised to INT8.
     * Pass-through prevents the runtime from quantising them again.
     */
    input_attr.pass_through = 1;
    output_attr.pass_through = 1;

    check_rknn(
        rknn_set_io_mem(ctx, input_mem, &input_attr),
        "set input memory"
    );

    check_rknn(
        rknn_set_io_mem(ctx, output_mem, &output_attr),
        "set output memory"
    );

    int class_correct[NUM_CLASSES] = {0};
    int class_total[NUM_CLASSES] = {0};
    int confusion[NUM_CLASSES][NUM_CLASSES] = {{0}};

    int total_correct = 0;
    int invalid_labels = 0;

    int8_t *input_buffer = (int8_t *)input_mem->virt_addr;
    int8_t *output_buffer = (int8_t *)output_mem->virt_addr;

    if (!input_buffer || !output_buffer)
        fail("RKNN tensor memory has a null virtual address");

    printf("\nStarting inference...\n");

    for (int sample = 0; sample < NUM_SAMPLES; sample++) {

        if (labels[sample] < 0 || labels[sample] >= NUM_CLASSES) {
            invalid_labels++;
            continue;
        }

        const float *sample_features =
            features + (size_t)sample * NUM_FEATURES;

        /* Quantise the 131 input features. */
        for (int j = 0; j < NUM_FEATURES; j++) {
            input_buffer[j] = quantize_input(sample_features[j]);
        }

        /* Run inference. */
        check_rknn(
            rknn_run(ctx, NULL),
            "rknn_run"
        );

        /* Find the highest output score. */
        int predicted = 0;
        int8_t highest = output_buffer[0];

        for (int j = 1; j < NUM_CLASSES; j++) {
            if (output_buffer[j] > highest) {
                highest = output_buffer[j];
                predicted = j;
            }
        }

        int actual = labels[sample];

        class_total[actual]++;
        confusion[actual][predicted]++;

        if (predicted == actual) {
            total_correct++;
            class_correct[actual]++;
        }

        if ((sample + 1) % 500 == 0 ||
            sample + 1 == NUM_SAMPLES) {
            printf("Processed %d / %d samples\n",
                   sample + 1, NUM_SAMPLES);
            fflush(stdout);
        }
    }

    /* Print summary. */
    printf("\n========================================\n");
    printf(" VALIDATION RESULTS\n");
    printf("========================================\n");

    int evaluated = NUM_SAMPLES - invalid_labels;

    printf("Correct:   %d\n", total_correct);
    printf("Incorrect: %d\n", evaluated - total_correct);
    printf("Evaluated: %d\n", evaluated);

    if (evaluated > 0) {
        printf("Accuracy:  %.4f%%\n",
               100.0 * total_correct / evaluated);
    }

    if (invalid_labels > 0)
        printf("Invalid labels skipped: %d\n", invalid_labels);

    printf("\nPer-class accuracy:\n");
    printf("%-8s %10s %10s %12s\n",
           "Class", "Correct", "Total", "Accuracy");

    for (int c = 0; c < NUM_CLASSES; c++) {
        double accuracy = class_total[c] > 0
            ? 100.0 * class_correct[c] / class_total[c]
            : 0.0;

        printf("%-8s %10d %10d %11.2f%%\n",
               default_class_names[c],
               class_correct[c],
               class_total[c],
               accuracy);
    }

    printf("\nConfusion matrix (actual rows, predicted columns):\n");
    printf("Actual\\Pred");
    for (int c = 0; c < NUM_CLASSES; c++)
        printf(",%s", default_class_names[c]);
    printf("\n");

    for (int r = 0; r < NUM_CLASSES; r++) {
        printf("%s", default_class_names[r]);

        for (int c = 0; c < NUM_CLASSES; c++)
            printf(",%d", confusion[r][c]);

        printf("\n");
    }

    /* Clean up. */
    check_rknn(rknn_destroy_mem(ctx, input_mem), "destroy input memory");
    check_rknn(rknn_destroy_mem(ctx, output_mem), "destroy output memory");
    check_rknn(rknn_destroy(ctx), "rknn_destroy");

    free(features);
    free(labels);

    printf("\nValidation complete.\n");
    return 0;
}
