# Contribuyendo a Zen C

¡Primero que nada, gracias por considerar contribuir a Zen C! Son personas como tú las que hacen grande a este proyecto.

Damos la bienvenida a todas las contribuciones, ya sea arreglando bugs, añadiendo documentación, proponiendo nuevas características, o simplemente reportando problemas.

## Cómo Contribuir

El flujo de trabajo general para contribuir es:

1.  **Haz un Fork del Repositorio**: Usa el flujo de trabajo estándar de GitHub para hacer un fork del repositorio a tu propia cuenta.
2.  **Crea una Rama de Característica**: Crea una nueva rama para tu característica o corrección de errores. Esto mantiene tus cambios organizados y separados de la rama principal.
    ```bash
    git checkout -b feature/NuevaCosa
    ```
3.  **Haz Cambios**: Escribe tu código o cambios de documentación.
4.  **Verifica**: Asegúrate de que tus cambios funcionen como se espera y no rompan la funcionalidad existente (ver [Ejecutando Pruebas](#ejecutando-pruebas)).
5.  **Envía un Pull Request**: Empuja tu rama a tu fork y envía un Pull Request (PR) al repositorio principal de Zen C.

## Problemas (Issues) y Solicitudes de Extracción (Pull Requests)

Usamos GitHub Issues y Pull Requests para rastrear errores y características. Para ayudarnos a mantener la calidad:

-   **Use Plantillas**: Al abrir un Issue o PR, use las plantillas proporcionadas.
    -   **Reporte de Error**: Para reportar errores.
    -   **Solicitud de Función**: Para sugerir nuevas características.
    -   **Solicitud de Extracción**: Para enviar cambios de código.
-   **Sea Descriptivo**: Proporcione tantos detalles como sea posible.
    -   **Comprobaciones Automatizadas**: Tenemos un flujo de trabajo automatizado que verifica la longitud de la descripción de nuevos Issues y PRs. Si la descripción es demasiado corta (< 50 caracteres), se cerrará automáticamente. Esto es para asegurar que tenemos suficiente información para ayudarle.

## Guías de Desarrollo

### Estilo de Código
- Sigue el estilo de C existente que se encuentra en el código base. La consistencia es clave.
- Puedes usar el archivo `.clang-format` proporcionado para formatear tu código.
- Mantén el código limpio y legible.

### Estructura del Proyecto
Si estás buscando extender el compilador, aquí hay un mapa rápido del código base:
*   **Parser**: `src/parser/` - Contiene la implementación del parser de descenso recursivo.
*   **Codegen**: `src/codegen/` - Contiene la lógica del transpilador que convierte Zen C a GNU C/C11.
*   **Biblioteca Estándar**: `std/` - Los módulos de la biblioteca estándar, escritos en el propio Zen C.

## Ejecutando Pruebas

La suite de pruebas es tu mejor amiga al desarrollar. Por favor, asegúrate de que todas las pruebas pasen antes de enviar un PR.

### Ejecutar Todas las Pruebas
Para ejecutar la suite de pruebas completa usando el compilador por defecto (usualmente GCC):
```bash
make test
```

### Ejecutar Prueba Específica
Para ejecutar un solo archivo de prueba para ahorrar tiempo durante el desarrollo:
```bash
./zc run tests/test_match.zc
```

### Probar con Diferentes Backends
Zen C soporta múltiples compiladores de C como backends. Puedes ejecutar pruebas contra ellos específicamente:

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

## Proceso de Pull Request

1.  Asegúrate de haber añadido pruebas para cualquier nueva funcionalidad.
2.  Asegúrate de que todas las pruebas existentes pasen.
3.  Actualiza la documentación (archivos Markdown en `docs/`, `translations/` o `README.md`) si es apropiado.
4.  Describe tus cambios claramente en la descripción del PR. Enlaza a cualquier issue relacionado.

¡Gracias por tu contribución!
