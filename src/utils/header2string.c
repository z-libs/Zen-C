#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Improved: Handle both slash types on all platforms for robustness
const char *get_basename(const char *path) {
    const char *last = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/' || *p == '\\') {
            last = p + 1;
        }
    }
    return last;
}

// Improved: Ensure the result is a valid C identifier
void sanitize_varname(char *dst, const char *src) {
    // C variables cannot start with a digit
    if (isdigit((unsigned char)*src)) {
        *dst++ = '_';
    }

    while (*src) {
        if (isalnum((unsigned char)*src)) {
            *dst = *src;
        } else {
            *dst = '_'; // Replace spaces, dots, colons, slashes, etc.
        }
        dst++;
        src++;
    }
    *dst = 0;
}

void get_stem(char *dst, const char *filename) {
    const char *dot = strrchr(filename, '.');
    size_t len = dot ? (size_t)(dot - filename) : strlen(filename);
    strncpy(dst, filename, len);
    dst[len] = 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <input_file> [output_header]\n", argv[0]);
        return 1;
    }

    const char *input_file = argv[1];
    char header_file[512];
    
    if (argc == 3) {
        strncpy(header_file, argv[2], sizeof(header_file) - 1);
        header_file[sizeof(header_file) - 1] = 0;
    } else {
        strncpy(header_file, input_file, sizeof(header_file) - 1);
        header_file[sizeof(header_file) - 1] = 0;
        char *dot = strrchr(header_file, '.');
        if (dot) strcpy(dot, ".h");
        else strcat(header_file, ".h");
    }

    FILE *in = fopen(input_file, "r");
    if (!in) {
        fprintf(stderr, "Error: cannot open %s\n", input_file);
        return 1;
    }
    FILE *out = fopen(header_file, "w");
    if (!out) {
        fclose(in);
        fprintf(stderr, "Error: cannot write %s\n", header_file);
        return 1;
    }

    char stem[256], varname[256];
    const char *base = get_basename(input_file);
    get_stem(stem, base);
    sanitize_varname(varname, stem);

    fprintf(out, "// Auto-generated header\n");
    fprintf(out, "#ifndef _EMBEDDED_%s_H_\n", varname);
    fprintf(out, "#define _EMBEDDED_%s_H_\n\n", varname);
    fprintf(out, "const char *%s_str =\n", varname);

    char line[1024];
    while (fgets(line, sizeof(line), in)) {
        fprintf(out, "    \"");
        for (char *src = line; *src; ++src) {
            if (*src == '\\') fprintf(out, "\\\\");
            else if (*src == '"') fprintf(out, "\\\"");
            else if (*src == '\n') fprintf(out, "\\n");
            else if (*src == '\r') continue;
            else fprintf(out, "%c", *src);
        }
        fprintf(out, "\"\n");
    }
    fprintf(out, "\"\"\n");
    fprintf(out, ";\n\n#endif\n\n");

    fclose(in);
    fclose(out);
    printf("Generated: %s (variable: %s_str)\n", header_file, varname);
    return 0;
}