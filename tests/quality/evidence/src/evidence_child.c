#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    if ((argc != 3 && argc != 4) || (argc == 4 && strcmp(argv[3], "--fail") != 0)) {
        fputs("usage: BoundedEvidenceChild input counter [--fail]\n", stderr);
        return 2;
    }

    FILE* input = fopen(argv[1], "rb");
    if (input == NULL) {
        fputs("could not open the declared input\n", stderr);
        return 3;
    }
    int value = 0;
    const int matched = fscanf(input, "value=%d", &value);
    const int input_closed = fclose(input);
    if (matched != 1 || value != 7 || input_closed != 0) {
        fputs("the declared input did not provide value=7\n", stderr);
        return 4;
    }

    if (printf("result=%d\n", value * 2) < 0 || fflush(stdout) != 0) {
        return 5;
    }
    if (argc == 4) {
        return 7;
    }

    FILE* counter = fopen(argv[2], "ab");
    if (counter == NULL) {
        fputs("could not open the declared counter output\n", stderr);
        return 6;
    }
    const int written = fputs("1\n", counter);
    const int counter_closed = fclose(counter);
    if (written == EOF || counter_closed != 0) {
        fputs("could not append the assertion counter\n", stderr);
        return 6;
    }
    return 0;
}
