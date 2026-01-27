#ifndef LSP_PROJECT_H
#define LSP_PROJECT_H

#include "parser.h"
#include "lsp_index.h"

/**
 * @brief Represents a tracked file in the LSP project.
 */
typedef struct ProjectFile
{
    char *path;      ///< Absolute file path.
    char *uri;       ///< file:// URI.
    char *source;    ///< Cached source content (in-memory).
    LSPIndex *index; ///< File-specific symbol index.
    struct ProjectFile *next;
} ProjectFile;

/**
 * @brief Global state for the Language Server Project.
 */
typedef struct
{
    /**
     * @brief Global shared parser context.
     * Contains global registries (structs, functions) reused across files.
     */
    ParserContext *ctx;

    ProjectFile *files; ///< List of tracked open files.
    char *root_path;    ///< Project root directory.
} LSPProject;

// Global project instance
extern LSPProject *g_project;

// Initialize the project with a root directory
void lsp_project_init(const char *root_path);

// Find a file in the project
ProjectFile *lsp_project_get_file(const char *uri);

// Update a file (re-parse and re-index)
void lsp_project_update_file(const char *uri, const char *src);

// Find definition globally
typedef struct
{
    char *uri;
    LSPRange *range;
} DefinitionResult;

DefinitionResult lsp_project_find_definition(const char *name);

typedef struct ReferenceResult
{
    char *uri;
    LSPRange *range;
    struct ReferenceResult *next;
} ReferenceResult;

ReferenceResult *lsp_project_find_references(const char *name);

#endif
