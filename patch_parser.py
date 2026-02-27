import re

with open("src/parser/parser_core.c", "r") as f:
    text = f.read()

# 1. int cfg_skip = 0; -> char *cfg_condition = NULL;
text = text.replace("int cfg_skip = 0;         // @cfg() conditional compilation", "char *cfg_condition = NULL; // @cfg() conditional compilation")

# 2. @cfg(not(NAME))
t1_old = """                        char *cfg_name = token_strdup(name_tok);
                        if (is_cfg_defined(cfg_name))
                        {
                            cfg_skip = 1;
                        }"""
t1_new = """                        char *cfg_name = token_strdup(name_tok);
                        if (!cfg_condition) {
                            cfg_condition = xmalloc(strlen(cfg_name) + 32);
                            sprintf(cfg_condition, "!defined(%s)", cfg_name);
                        } else {
                            char *old = cfg_condition;
                            cfg_condition = xmalloc(strlen(old) + strlen(cfg_name) + 32);
                            sprintf(cfg_condition, "%s && !defined(%s)", old, cfg_name);
                            free(old);
                        }
                        free(cfg_name);"""
text = text.replace(t1_old, t1_new)

# 3. @cfg(any(...))
t2_old = """                        int any_match = 0;
                        while (1)
                        {
                            Token t = lexer_next(l);
                            if (t.type == TOK_IDENT && t.len == 3 &&
                                strncmp(t.start, "not", 3) == 0)
                            {
                                if (lexer_next(l).type != TOK_LPAREN)
                                {
                                    zpanic_at(lexer_peek(l), "Expected ( after not");
                                }
                                Token nt = lexer_next(l);
                                if (nt.type != TOK_IDENT)
                                {
                                    zpanic_at(nt, "Expected define name");
                                }
                                if (!is_cfg_defined(token_strdup(nt)))
                                {
                                    any_match = 1;
                                }
                                if (lexer_next(l).type != TOK_RPAREN)
                                {
                                    zpanic_at(lexer_peek(l), "Expected )");
                                }
                            }
                            else if (t.type == TOK_IDENT)
                            {
                                if (is_cfg_defined(token_strdup(t)))
                                {
                                    any_match = 1;
                                }
                            }
                            else
                            {
                                zpanic_at(t, "Expected define name in @cfg(any(...))");
                            }
                            if (lexer_peek(l).type == TOK_COMMA)
                            {
                                lexer_next(l);
                            }
                            else
                            {
                                break;
                            }
                        }
                        if (lexer_next(l).type != TOK_RPAREN)
                        {
                            zpanic_at(lexer_peek(l), "Expected ) after any(...)");
                        }
                        if (!any_match)
                        {
                            cfg_skip = 1;
                        }"""
t2_new = """                        char *any_cond = NULL;
                        while (1)
                        {
                            Token t = lexer_next(l);
                            if (t.type == TOK_IDENT && t.len == 3 &&
                                strncmp(t.start, "not", 3) == 0)
                            {
                                if (lexer_next(l).type != TOK_LPAREN)
                                {
                                    zpanic_at(lexer_peek(l), "Expected ( after not");
                                }
                                Token nt = lexer_next(l);
                                if (nt.type != TOK_IDENT)
                                {
                                    zpanic_at(nt, "Expected define name");
                                }
                                char *cfg_name = token_strdup(nt);
                                if (!any_cond) {
                                    any_cond = xmalloc(strlen(cfg_name) + 32);
                                    sprintf(any_cond, "!defined(%s)", cfg_name);
                                } else {
                                    char *old = any_cond;
                                    any_cond = xmalloc(strlen(old) + strlen(cfg_name) + 32);
                                    sprintf(any_cond, "%s || !defined(%s)", old, cfg_name);
                                    free(old);
                                }
                                free(cfg_name);
                                if (lexer_next(l).type != TOK_RPAREN)
                                {
                                    zpanic_at(lexer_peek(l), "Expected )");
                                }
                            }
                            else if (t.type == TOK_IDENT)
                            {
                                char *cfg_name = token_strdup(t);
                                if (!any_cond) {
                                    any_cond = xmalloc(strlen(cfg_name) + 32);
                                    sprintf(any_cond, "defined(%s)", cfg_name);
                                } else {
                                    char *old = any_cond;
                                    any_cond = xmalloc(strlen(old) + strlen(cfg_name) + 32);
                                    sprintf(any_cond, "%s || defined(%s)", old, cfg_name);
                                    free(old);
                                }
                                free(cfg_name);
                            }
                            else
                            {
                                zpanic_at(t, "Expected define name in @cfg(any(...))");
                            }
                            if (lexer_peek(l).type == TOK_COMMA)
                            {
                                lexer_next(l);
                            }
                            else
                            {
                                break;
                            }
                        }
                        if (lexer_next(l).type != TOK_RPAREN)
                        {
                            zpanic_at(lexer_peek(l), "Expected ) after any(...)");
                        }
                        if (any_cond) {
                            if (!cfg_condition) {
                                cfg_condition = xmalloc(strlen(any_cond) + 32);
                                sprintf(cfg_condition, "(%s)", any_cond);
                            } else {
                                char *old = cfg_condition;
                                cfg_condition = xmalloc(strlen(old) + strlen(any_cond) + 32);
                                sprintf(cfg_condition, "%s && (%s)", old, any_cond);
                                free(old);
                            }
                            free(any_cond);
                        }"""
text = text.replace(t2_old, t2_new)

# 4. @cfg(all(...))
t3_old = """                        int all_match = 1;
                        while (1)
                        {
                            Token t = lexer_next(l);
                            if (t.type == TOK_IDENT && t.len == 3 &&
                                strncmp(t.start, "not", 3) == 0)
                            {
                                if (lexer_next(l).type != TOK_LPAREN)
                                {
                                    zpanic_at(lexer_peek(l), "Expected ( after not");
                                }
                                Token nt = lexer_next(l);
                                if (nt.type != TOK_IDENT)
                                {
                                    zpanic_at(nt, "Expected define name");
                                }
                                if (is_cfg_defined(token_strdup(nt)))
                                {
                                    all_match = 0;
                                }
                                if (lexer_next(l).type != TOK_RPAREN)
                                {
                                    zpanic_at(lexer_peek(l), "Expected )");
                                }
                            }
                            else if (t.type == TOK_IDENT)
                            {
                                if (!is_cfg_defined(token_strdup(t)))
                                {
                                    all_match = 0;
                                }
                            }
                            else
                            {
                                zpanic_at(t, "Expected define name in @cfg(all(...))");
                            }
                            if (lexer_peek(l).type == TOK_COMMA)
                            {
                                lexer_next(l);
                            }
                            else
                            {
                                break;
                            }
                        }
                        if (lexer_next(l).type != TOK_RPAREN)
                        {
                            zpanic_at(lexer_peek(l), "Expected ) after all(...)");
                        }
                        if (!all_match)
                        {
                            cfg_skip = 1;
                        }"""
t3_new = """                        char *all_cond = NULL;
                        while (1)
                        {
                            Token t = lexer_next(l);
                            if (t.type == TOK_IDENT && t.len == 3 &&
                                strncmp(t.start, "not", 3) == 0)
                            {
                                if (lexer_next(l).type != TOK_LPAREN)
                                {
                                    zpanic_at(lexer_peek(l), "Expected ( after not");
                                }
                                Token nt = lexer_next(l);
                                if (nt.type != TOK_IDENT)
                                {
                                    zpanic_at(nt, "Expected define name");
                                }
                                char *cfg_name = token_strdup(nt);
                                if (!all_cond) {
                                    all_cond = xmalloc(strlen(cfg_name) + 32);
                                    sprintf(all_cond, "!defined(%s)", cfg_name);
                                } else {
                                    char *old = all_cond;
                                    all_cond = xmalloc(strlen(old) + strlen(cfg_name) + 32);
                                    sprintf(all_cond, "%s && !defined(%s)", old, cfg_name);
                                    free(old);
                                }
                                free(cfg_name);
                                if (lexer_next(l).type != TOK_RPAREN)
                                {
                                    zpanic_at(lexer_peek(l), "Expected )");
                                }
                            }
                            else if (t.type == TOK_IDENT)
                            {
                                char *cfg_name = token_strdup(t);
                                if (!all_cond) {
                                    all_cond = xmalloc(strlen(cfg_name) + 32);
                                    sprintf(all_cond, "defined(%s)", cfg_name);
                                } else {
                                    char *old = all_cond;
                                    all_cond = xmalloc(strlen(old) + strlen(cfg_name) + 32);
                                    sprintf(all_cond, "%s && defined(%s)", old, cfg_name);
                                    free(old);
                                }
                                free(cfg_name);
                            }
                            else
                            {
                                zpanic_at(t, "Expected define name in @cfg(all(...))");
                            }
                            if (lexer_peek(l).type == TOK_COMMA)
                            {
                                lexer_next(l);
                            }
                            else
                            {
                                break;
                            }
                        }
                        if (lexer_next(l).type != TOK_RPAREN)
                        {
                            zpanic_at(lexer_peek(l), "Expected ) after all(...)");
                        }
                        if (all_cond) {
                            if (!cfg_condition) {
                                cfg_condition = xmalloc(strlen(all_cond) + 32);
                                sprintf(cfg_condition, "(%s)", all_cond);
                            } else {
                                char *old = cfg_condition;
                                cfg_condition = xmalloc(strlen(old) + strlen(all_cond) + 32);
                                sprintf(cfg_condition, "%s && (%s)", old, all_cond);
                                free(old);
                            }
                            free(all_cond);
                        }"""
text = text.replace(t3_old, t3_new)

# 5. @cfg(NAME)
t4_old = """                        char *cfg_name = token_strdup(cfg_tok);
                        if (!is_cfg_defined(cfg_name))
                        {
                            cfg_skip = 1; // Not defined — skip
                        }"""
t4_new = """                        char *cfg_name = token_strdup(cfg_tok);
                        if (!cfg_condition) {
                            cfg_condition = xmalloc(strlen(cfg_name) + 32);
                            sprintf(cfg_condition, "defined(%s)", cfg_name);
                        } else {
                            char *old = cfg_condition;
                            cfg_condition = xmalloc(strlen(old) + strlen(cfg_name) + 32);
                            sprintf(cfg_condition, "%s && defined(%s)", old, cfg_name);
                            free(old);
                        }
                        free(cfg_name);"""
text = text.replace(t4_old, t4_new)

# 6. Removing: if (cfg_skip) { skip_top_level_decl(l); continue; }
t5_old = """        if (cfg_skip)
        {
            skip_top_level_decl(l);
            continue;
        }"""
t5_new = """        // Removed cfg_skip handling here"""
text = text.replace(t5_old, t5_new)

# 7. Add cfg assignment at the end of the loop
t6_old = """        if (s)
        {
            if (!h)"""
t6_new = """        if (s)
        {
            s->cfg_condition = cfg_condition;

            if (!h)"""

t7_old = """            while (tl->next)
            {
                tl = tl->next;
            }
        }
    }"""
t7_new = """            while (tl->next)
            {
                tl = tl->next;
            }
        }
        else if (cfg_condition)
        {
            free(cfg_condition);
        }
    }"""
text = text.replace(t6_old, t6_new)
text = text.replace(t7_old, t7_new)

with open("src/parser/parser_core.c", "w") as f:
    f.write(text)

