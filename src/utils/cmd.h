#ifndef CMD_H
#define CMD_H

#include <stddef.h>

typedef struct CompilerConfig CompilerConfig;

typedef struct
{
    char *buf;
    size_t len;
    size_t cap;
} CmdBuilder;

typedef struct
{
    char **args;
    size_t count;
    size_t cap;
} ArgList;

/**
 * @brief Initialize a new command builder
 * @param cmd The command builder to initialize
 */
void cmd_init(CmdBuilder *cmd);

/**
 * @brief Add a string to the command (appends space if needed)
 * @param cmd The command builder to add to
 * @param str The string to add
 */
void cmd_add(CmdBuilder *cmd, const char *str);

/**
 * @brief Add a format string to the command
 * @param cmd The command builder to add to
 * @param fmt The format string
 * @param ... The arguments to format
 */
void cmd_add_fmt(CmdBuilder *cmd, const char *fmt, ...);

/**
 * @brief Free the command builder
 * @param cmd The command builder to free
 */
void cmd_free(CmdBuilder *cmd);

/**
 * @brief Get the built command as a string (owned by the builder)
 * @param cmd The command builder to get the string from
 * @return The built command as a string
 */
const char *cmd_to_string(CmdBuilder *cmd);

/**
 * @brief Print compiler library search paths
 */
void print_search_paths(CompilerConfig *cfg);

/**
 * @brief Print compiler version
 */
void print_version();

/**
 * @brief Print compiler usage string
 */
void print_usage();
void print_command_help(const char *command);

/**
 * @brief Initialize a new argument list
 * @param list The list to initialize
 */
void arg_list_init(ArgList *list);

/**
 * @brief Add an argument to the list
 * @param list The list to add to
 * @param arg The argument to add (will be duplicated)
 */
void arg_list_add(ArgList *list, const char *arg);

/**
 * @brief Add a formatted argument to the list
 * @param list The list to add to
 * @param fmt The format string
 * @param ... The arguments to format
 */
void arg_list_add_fmt(ArgList *list, const char *fmt, ...);

/**
 * @brief Free the argument list
 * @param list The list to free
 */
void arg_list_free(ArgList *list);

/**
 * @brief Run the argument list securely
 * @param list The list to run
 * @return Exit code
 */
int arg_run(ArgList *list);

/**
 * @brief Add arguments from a space-separated string to the list
 * @param list The list to add to
 * @param str The string to parse
 */
void arg_list_add_from_string(ArgList *list, const char *str);

void build_compile_arg_list(ArgList *list, const char *outfile, const char *temp_source_file,
                            CompilerConfig *cfg);

#endif
