from docling.document_converter import DocumentConverter

def extract_data_dictionary(url: str):
    tables = DocumentConverter().convert(url).document.export_to_dict().get('tables', [])

    colunas, caracteristicas, tipos = [], [], []
    tipos_colunas = ['varchar', 'date', 'number']

    for table in tables:
        for conteudo in table['data']['table_cells']:
            texto = conteudo.get('text', '')
            if '_' in texto:
                colunas.append(texto)
            elif tipos_colunas[0] in texto.lower() or tipos_colunas[1] in texto.lower() or tipos_colunas[2] in texto.lower():
                tipos.append(texto)
            elif 'campo' in texto.lower():
                caracteristicas.append(texto)

    structs = {}
    for i in range(len(colunas)):
        structs[colunas[i]] = {
            'tipo': tipos[i] if i < len(tipos) else None,
            'caracteristicas': caracteristicas[i] if i < len(caracteristicas) else None
        }

    return structs