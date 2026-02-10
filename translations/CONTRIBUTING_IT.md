# Contribuisci a Zen C

Innanzitutto, grazie per aver preso in considerazione di contribuire a Zen C! Sono le persone come te che rendono grande questo progetto.

Accogliamo tutti i contributi, che siano fix di bug, miglioramenti alla documentazione, la proposta di nuove funzionalità, o semplicemente la segnalazione di problemi.

## Come contribuire

Il flusso di lavoro generale per contribuire è:

1.  **Forka la repository**: Usa il workflow standard di GitHub per fare il fork della repository sul tuo account.
2.  **Crea un Branch per la funzionalità**: Crea un nuovo branch per la tua funzionalità o fix di bug. Questo mantiene le tue modifiche organizzate e separate dal branch principale.
    ```bash
    git checkout -b feature/NewThing
    ```
3.  **Apporta Modifiche**: Scrivi il tuo codice o le modifiche alla documentazione.
4.  **Verifica**: Assicurati che le tue modifiche funzionino come previsto e non rompano le funzionalità esistenti (vedi [Eseguire i test](#eseguire-i-test)).
5.  **Crea una Pull Request**: Pusha il tuo branch sul tuo fork e invia una Pull Request (PR) alla repository principale di Zen C.

## Segnalazioni (Issues) e Richieste di Pull (Pull Requests)

Usiamo le GitHub Issues e le Pull Requests per tracciare bug e funzionalità. Per aiutarci a mantenere la qualità:

-   **Usa i Modelli**: Quando apri una Issue o una PR, usa i modelli forniti.
    -   **Segnalazione Bug**: Per segnalare errori.
    -   **Richiesta Funzionalità**: Per suggerire nuove funzionalità.
    -   **Richiesta di Pull**: Per inviare modifiche al codice.
-   **Sii Descrittivo**: Fornire quanti più dettagli possibile.
    -   **Controlli Automatici**: Abbiamo un flusso di lavoro automatizzato che controlla la lunghezza della descrizione delle nuove Issues e PR. Se la descrizione è troppo breve (< 50 caratteri), verrà chiusa automaticamente. Questo serve per assicurarci di avere abbastanza informazioni per aiutarti.

## Guide Linea per il Sviluppo

### Stile del Codice
- Segui lo stile C esistente trovato nel codebase. La coerenza è fondamentale.
- Puoi usare il file `.clang-format` fornito per formattare il tuo codice.
- Mantieni il codice pulito e leggibile.

### Struttura del Progetto
Se stai cercando di estendere il compilatore, ecco una mappa rapida del codebase:
*   **Parser**: `src/parser/` - Contiene l'implementazione del parser a discesa ricorsiva.
*   **Codegen**: `src/codegen/` - Contiene la logica del transpiler che converte Zen C a GNU C/C11.
*   **Standard Library**: `std/` - I moduli della libreria standard, scritti nello stesso Zen C.

## Eseguire i test

La suite di test è il tuo miglior amico durante lo sviluppo. Assicurati che passino tutti i test prima di inviare una PR.

### Esegui tutti i test
Per eseguire l'intera suite di test usando il compilatore predefinito (solitamente GCC):
```bash
make test
```

### Esegui test specifici
Per eseguire un singolo file di test per risparmiare tempo durante lo sviluppo:
```bash
./zc run tests/test_match.zc
```

### Test con Backend Differenti
Zen C supporta diversi compilatori C come backend. Puoi eseguire i test specificamente contro di essi:

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

## Processo di Pull Request

1.  Assicurati di aver aggiunto test per qualsiasi nuova funzionalità.
2.  Assicurati che tutti i test esistenti passino.
3.  Aggiorna la documentazione (file Markdown in `docs/`, `translations/` o `README.md`) se appropriato.
4.  Descrivi chiaramente le tue modifiche nella descrizione della PR. Linka qualsiasi issue correlata.

Grazie per il tuo contributo!
