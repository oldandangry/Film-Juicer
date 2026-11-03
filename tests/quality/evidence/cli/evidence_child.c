#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
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
    char line[64];
    const char* read = fgets(line, sizeof(line), input);
    const int extra = read != NULL && ferror(input) == 0 && feof(input) == 0 ? fgetc(input) : EOF;
    const int input_failed = ferror(input);
    const int input_closed = fclose(input);
    char* end = NULL;
    long parsed = 0;
    errno = 0;
    if (read != NULL && strncmp(line, "value=", 6) == 0) {
        parsed = strtol(line + 6, &end, 10);
    }
    const int suffix_valid = end != NULL && (*end == '\0' || (*end == '\n' && end[1] == '\0'));
    if (read == NULL || end == NULL || end == line + 6 || errno != 0 || !suffix_valid || parsed < INT_MIN || parsed > INT_MAX || parsed != 7 || extra != EOF || input_failed != 0 || input_closed != 0) {
        fputs("the declared input did not provide value=7\n", stderr);
        return 4;
    }
    const int value = (int)parsed;
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
