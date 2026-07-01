# llm_book_automata
Creates books using small LLM models thru LMstudio, publishes every book

## Pasos de procesamiento

- User: sube su takeout
- System: preprocesa takeout
- User: desselecciona datos que no quiere que sean procesados
- System: crea un json de los datos seleccionados finales
- LLM : parsea una ficha de usuario con:
    - Nombre apellidos
    - Género
    - Edad
    - Lugar de nacimiento
    - Intereses []


## Modelos recomendados