# Contribuindo para Zen C

Primeiramente, obrigado por considerar contribuir para o Zen C! São pessoas como você que tornam este projeto ótimo.

Nós damos boas-vindas a todas as contribuições, seja consertando bugs, adicionando documentação, propondo novas funcionalidades ou apenas reportando problemas.

## Como Contribuir

O fluxo de trabalho geral para contribuir é:

1.  **Fork do Repositório**: Use o fluxo de trabalho padrão do GitHub para fazer um fork do repositório para sua própria conta.
2.  **Crie um Feature Branch**: Crie um novo branch para sua funcionalidade ou correção de bug. Isso mantém suas mudanças organizadas e separadas do branch principal.
    ```bash
    git checkout -b feature/NewThing
    ```
3.  **Faça Mudanças**: Escreva seu código ou mudanças de documentação.
4.  **Verifique**: Garanta que suas mudanças funcionem como esperado e não quebrem funcionalidades existentes (veja [Executando Testes](#executando-testes)).
5.  **Submeta um Pull Request**: Dê push do seu branch para seu fork e submeta um Pull Request (PR) para o repositório principal do Zen C.

## Problemas (Issues) e Pull Requests

Usamos GitHub Issues e Pull Requests para rastrear bugs e recursos. Para nos ajudar a manter a qualidade:

-   **Use Templates**: Ao abrir uma Issue ou PR, use os templates fornecidos.
    -   **Relatório de Bug**: Para reportar erros.
    -   **Solicitação de Recurso**: Para sugerir novos recursos.
    -   **Pull Request**: Para enviar alterações de código.
-   **Seja Descritivo**: Forneça o máximo de detalhes possível.
    -   **Verificações Automatizadas**: Temos um fluxo de trabalho automatizado que verifica o comprimento da descrição de novas Issues e PRs. Se a descrição for muito curta (< 50 caracteres), ela será fechada automaticamente. Isso é para garantir que tenhamos informações suficientes para ajudá-lo.

## Diretrizes de Desenvolvimento

### Estilo de Código
- Siga o estilo C existente encontrado na base de código. Consistência é chave.
- Você pode usar o arquivo `.clang-format` fornecido para formatar seu código.
- Mantenha o código limpo e legível.

### Estrutura do Projeto
Se você está procurando estender o compilador, aqui está um mapa rápido da base de código:
*   **Parser**: `src/parser/` - Contém a implementação do parser de descida recursiva.
*   **Codegen**: `src/codegen/` - Contém a lógica do transpilador que converte Zen C para GNU C/C11.
*   **Biblioteca Padrão**: `std/` - Os módulos da biblioteca padrão, escritos no próprio Zen C.

## Executando Testes

A suíte de testes é sua melhor amiga ao desenvolver. Por favor, garanta que todos os testes passem antes de submeter um PR.

### Executar Todos os Testes
Para executar a suíte de testes completa usando o compilador padrão (geralmente GCC):
```bash
make test
```

### Executar Teste Específico
Para executar um único arquivo de teste para economizar tempo durante o desenvolvimento:
```bash
./zc run tests/test_match.zc
```

### Testar com Backends Diferentes
Zen C suporta múltiplos compiladores C como backends. Você pode executar testes contra eles especificamente:

**Clang**:
```bash
./tests/run_tests.sh --cc clang
```

**Zig (cc)**:
```bash
./tests/run_tests.sh --cc zig
```

**TCC (Tiny C Compiler)**:
```bash
./tests/run_tests.sh --cc tcc
```

## Processo de Pull Request

1.  Garanta que você adicionou testes para qualquer nova funcionalidade.
2.  Garanta que todos os testes existentes passem.
3.  Atualize a documentação (arquivos Markdown em `docs/`, `translations/` ou `README.md`) se apropriado.
4.  Descreva suas mudanças claramente na descrição do PR. Link para qualquer issue relacionada.

Obrigado pela sua contribuição!
