# Exemplos de perguntas e respostas (execução real)

> **Origem das respostas.** Todas as respostas abaixo foram geradas **de verdade** pelo
> `pychat`, em 2026-09-23, usando o modelo `openai/gpt-oss-120b` (modelo aberto da OpenAI)
> servido pela API gratuita da **Groq** via `OPENAI_BASE_URL`. **Não** são respostas do
> GPT-4: o enunciado cita o GPT-4, o código o suporta trocando apenas `PYCHAT_MODEL`
> (e removendo `OPENAI_BASE_URL`), mas o desenvolvimento usou um provedor gratuito. O texto
> é reproduzido sem edição; por ser um LLM, uma nova execução produzirá variações.

Reproduza com `pychat ask --raw "<pergunta>"`. As perguntas estão em
[`examples/questions.txt`](../examples/questions.txt) e `make demo` roda todas.

Configuração usada (`.env`):

```text
OPENAI_API_KEY=<chave da Groq>
OPENAI_BASE_URL=https://api.groq.com/openai/v1
PYCHAT_MODEL=openai/gpt-oss-120b
```

## 1. A pergunta do enunciado

```bash
pychat ask --raw "Como criar uma lista em Python?"
```
<details><summary>Resposta completa (3143 caracteres)</summary>

**Resposta curta:**  
Em Python, uma lista é criada colocando os elementos entre colchetes `[]` ou usando a função `list()`.

**Explicação detalhada**

Uma lista (`list`) é uma coleção ordenada, mutável e indexada que pode armazenar itens de tipos diferentes.  
Algumas formas comuns de criá‑la:

| Método | Quando usar | Exemplo |
|--------|-------------|---------|
| **Colchetes literais** | Quando já conhece os valores iniciais. | `lista = [1, 2, 3]` |
| **`list()` a partir de outro iterável** | Quando tem um iterável (string, tupla, range, etc.) e quer convertê‑lo. | `lista = list('abc')` |
| **Compreensão de lista** | Quando precisa gerar a lista a partir de uma expressão ou filtro. | `lista = [x*x for x in range(5)]` |
| **Multiplicação de lista** | Para criar uma lista com valores repetidos (cuidado com objetos mutáveis). | `zeros = [0] * 5` |

### Exemplo completo (Python 3.12)

```python
# 1. Lista literal
numeros = [10, 20, 30]
print("Lista literal:", numeros)

# 2. Convertendo um iterável (string) em lista de caracteres
letras = list("Python")
print("Lista a partir de string:", letras)

# 3. Compreensão de lista: quadrados de 0 a 4
quadrados = [x**2 for x in range(5)]
print("Compreensão de lista:", quadrados)

# 4. Lista de valores repetidos
repetidos = ["x"] * 3
print("Repetidos:", repetidos)

# 5. Atenção ao usar objetos mutáveis como valores repetidos
listas_aninhadas = [[]] * 3
listas_aninhadas[0].append(1)
print("Listas aninhadas (efeito colateral):", listas_aninhadas)
```

**Saída esperada**

```
Lista literal: [10, 20, 30]
Lista a partir de string: ['P', 'y', 't', 'h', 'o', 'n']
Compreensão de lista: [0, 1, 4, 9, 16]
Repetidos: ['x', 'x', 'x']
Listas aninhadas (efeito colateral): [[1], [1], [1]]
```

### Por que o código funciona?

- **Colchetes** criam diretamente um objeto `list` contendo os itens especificados.
- `list(iterável)` itera sobre o argumento e coloca cada elemento na nova lista.
- **Compreensão de lista** (`[expressão for item in iterável]`) avalia a expressão para cada item, gerando uma nova lista de forma concisa.
- Multiplicar uma lista (`[valor] * n`) replica a referência ao mesmo objeto `valor` `n` vezes. Quando `valor` é imutável (como `0` ou `'x'`), isso é seguro. Quando é mutável (como `[]`), todas as posições apontam para o mesmo objeto, como demonstrado no último exemplo.

### Armadilhas comuns

1. **Referências compartilhadas** – ao usar `*` com objetos mutáveis, todas as posições referenciam o mesmo objeto. Use uma compreensão (`[[] for _ in range(3)]`) se precisar de listas independentes.
2. **Índices fora do intervalo** – acessar `lista[10]` em uma lista menor gera `IndexError`.
3. **Mistura de tipos** – Python permite, mas pode tornar o código menos legível; prefira manter tipos homogêneos quando possível.

### Alternativas idiomáticas

- **`deque`** da biblioteca `collections` quando precisar de inserções/remoções rápidas nas extremidades.
- **`array.array`** se todos os elementos forem numéricos do mesmo tipo e quiser economizar memória.

Com essas informações você já pode criar e manipular listas de forma segura e idiomática em Python. 🚀

</details>

## 2. Lista x tupla

```bash
pychat ask --raw "Qual a diferença entre lista e tupla?"
```
<details><summary>Resposta completa (4547 caracteres)</summary>

**Resposta curta:**  
A lista (`list`) é mutável – você pode alterar, inserir ou remover itens depois de criada.  
A tupla (`tuple`) é imutável – depois de criada seus elementos não podem ser modificados, o que a torna mais segura e, em alguns casos, mais rápida e utilizável como chave de dicionário.

---

## Explicação detalhada

| Característica | Lista (`list`) | Tupla (`tuple`) |
|----------------|----------------|-----------------|
| **Mutabilidade** | Mutável – pode mudar tamanho e conteúdo | Imutável – tamanho e conteúdo fixos |
| **Sintaxe** | `[]` ou `list()` | `()` ou `tuple()` |
| **Métodos** | Muitos (ex.: `append`, `extend`, `pop`, `remove`, `sort`, …) | Poucos (apenas `count` e `index`) |
| **Performance** | Levemente mais lenta para acesso e iteração devido ao overhead de mutabilidade | Levemente mais rápida e consome menos memória |
| **Hashability** | Não hashable (não pode ser chave de `dict` ou elemento de `set`) | Hashable se todos os elementos forem hashable – pode ser chave de `dict` ou elemento de `set` |
| **Uso típico** | Coleções que vão mudar ao longo da execução (filas, pilhas, buffers) | Dados “constantes”, registros, chaves de dicionário, retorno de múltiplos valores de funções |

### Por que a imutabilidade importa?

- **Segurança**: Quando você passa uma tupla para uma função, tem certeza de que o código da função não vai alterá‑la inadvertidamente.
- **Hashability**: Como a tupla não muda, o Python pode calcular um hash estável, permitindo seu uso como chave de dicionário ou elemento de `set`.
- **Otimizações internas**: O interpretador pode armazenar tuplas de forma mais compacta e reutilizá‑las (internamente, algumas tuplas são “interned”).

### Exemplo prático

```python
# Python 3.12
def demonstra_mutabilidade():
    # Lista mutável
    lista = [1, 2, 3]
    print("Lista original:", lista)

    lista.append(4)  # adiciona elemento
    lista[0] = 10  # altera elemento existente
    print("Lista após modificações:", lista)

    # Tupla imutável
    tupla = (1, 2, 3)
    print("\nTupla original:", tupla)

    try:
        tupla += (4,)  # cria uma nova tupla (não altera a original)
        print("Tupla após concatenação:", tupla)
        tupla[0] = 10  # tenta modificar -> erro
    except TypeError as e:
        print("Erro ao modificar tupla:", e)

    # Tupla como chave de dicionário
    mapa = {tupla: "valor associado"}
    print("\nAcesso ao dicionário usando a tupla como chave:", mapa[tupla])


if __name__ == "__main__":
    demonstra_mutabilidade()
```

**Saída esperada**

```
Lista original: [1, 2, 3]
Lista após modificações: [10, 2, 3, 4]

Tupla original: (1, 2, 3)
Tupla após concatenação: (1, 2, 3, 4)
Erro ao modificar tupla: 'tuple' object does not support item assignment

Acesso ao dicionário usando a tupla como chave: valor associado
```

#### O que o exemplo mostra?

1. **Listas podem ser alteradas in‑place** (`append`, atribuição por índice).  
2. **Tuplas não podem ser alteradas**; a tentativa de atribuir a um índice gera `TypeError`.  
3. **Concatenação (`+=`) com tupla** cria **uma nova tupla**, não modifica a original.  
4. **Tupla pode ser usada como chave** porque, após a concatenação, ainda é hashable.

### Armadilhas comuns

- **Confundir parênteses com tupla**: `x = (1)` não cria tupla, mas um inteiro entre parênteses. Use a vírgula: `x = (1,)`.
- **Mutabilidade de objetos internos**: Uma tupla pode conter objetos mutáveis (ex.: listas). Nesse caso, a tupla em si é imutável, mas seu conteúdo pode mudar, tornando‑a **não hashable**. Ex.:

  ```python
  t = ([1, 2], 3)  # t não pode ser usado como chave de dict
  ```

- **Performance enganosa**: A diferença de velocidade entre listas e tuplas costuma ser insignificante para a maioria dos programas; escolha a estrutura que melhor expressa a intenção do seu código.

### Quando usar cada uma?

- **Use `list`** quando precisar de:
  - Inserções/remissões frequentes.
  - Ordenação ou filtragem que modifica a coleção.
  - Estruturas de dados dinâmicas (pilhas, filas, buffers).

- **Use `tuple`** quando:
  - O conjunto de valores é fixo e não deve mudar.
  - Precisa de uma chave de dicionário ou elemento de `set`.
  - Quer deixar claro que a sequência representa um “registro” (ex.: coordenadas `(x, y, z)`).

---

**Resumo:** listas são coleções mutáveis e flexíveis, enquanto tuplas são coleções imutáveis, mais leves e hashable. Escolha a estrutura que melhor comunica a intenção do seu código e que atenda às necessidades de mutabilidade.

</details>

## 3. Ler arquivo linha por linha

```bash
pychat ask --raw "Como ler um arquivo linha por linha de forma eficiente?"
```
<details><summary>Resposta completa (4628 caracteres)</summary>

**Resposta curta:**  
Use o objeto de arquivo como um iterador dentro de um bloco `with`; assim o Python lê cada linha sob demanda, usando um buffer interno que evita carregar todo o conteúdo na memória.

---

## Por que essa abordagem é eficiente?

- **Iterador nativo:** Um objeto retornado por `open()` implementa o protocolo de iterador. Cada iteração (`for linha in f`) chama internamente `f.readline()` e devolve a próxima linha já decodificada.
- **Buffer interno:** O `open()` cria um buffer (por padrão 8 KB) que lê blocos do disco de uma só vez, reduzindo chamadas ao sistema.
- **Gerenciamento automático de recursos:** O `with` garante que o arquivo será fechado mesmo que ocorram exceções.
- **Baixo consumo de memória:** Apenas a linha corrente (e o buffer) permanecem na RAM, ideal para arquivos gigabytes.

### Código completo e executável (Python 3.12+)

```python
# exemplo.py
from pathlib import Path


def contar_linhas(path: Path) -> int:
    """Retorna a quantidade de linhas do arquivo, lendo uma a uma."""
    total = 0
    # O modo 'r' abre para leitura de texto; o encoding padrão é UTF‑8 a partir do 3.10.
    with path.open("r", encoding="utf-8") as f:
        for linha in f:  # iteração linha a linha
            total += 1
            # aqui você pode processar a linha, por exemplo:
            # print(linha.rstrip())
    return total


if __name__ == "__main__":
    # Cria um arquivo de teste grande (10 000 linhas) apenas para demonstração.
    teste = Path("arquivo_grande.txt")
    if not teste.exists():
        with teste.open("w", encoding="utf-8") as out:
            for i in range(1, 10_001):
                out.write(f"Linha {i}\n")

    print(f"Número total de linhas: {contar_linhas(teste)}")
```

**Saída esperada**

```
Número total de linhas: 10000
```

---

## Detalhes e variações úteis

| Estratégia | Quando usar | Como fazer | Observação |
|------------|-------------|------------|------------|
| `for linha in f:` | Leitura sequencial padrão | Como no exemplo acima | Mais legível e rápido que `while f.readline():` |
| `while (linha := f.readline()):` | Quando precisa de controle extra (ex.: pular linhas) | ```python\nwhile (linha := f.readline()):\n    ...\n``` | Disponível a partir do Python 3.8 (walrus operator). |
| `itertools.islice(f, start, stop)` | Pular cabeçalhos ou ler apenas um trecho | ```python\nfrom itertools import islice\nwith open('dados.csv') as f:\n    for linha in islice(f, 1, None):  # ignora a primeira linha\n        ...\n``` | Não carrega o arquivo inteiro. |
| `mmap.mmap` | Arquivos extremamente grandes onde se deseja buscar substrings sem ler linha a linha | ```python\nimport mmap\nwith open('gigante.log', 'r') as f:\n    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:\n        for linha in iter(mm.readline, b''):\n            ...\n``` | Útil para buscas rápidas, mas menos legível; requer tratamento de bytes. |
| `Path.read_text().splitlines()` | Arquivos pequenos (KB) onde a simplicidade vale mais que memória | ```python\nlinhas = Path('pequeno.txt').read_text().splitlines()\n``` | **Não** recomendado para arquivos grandes, pois carrega tudo na memória. |

### Armadilhas comuns

1. **Usar `readlines()`** – Carrega todas as linhas em uma lista, consumindo memória proporcional ao tamanho do arquivo.
2. **Esquecer de fechar o arquivo** – Pode deixar descritores abertos; o `with` resolve isso.
3. **Misturar modos binário e texto** – Se abrir com `'rb'` e iterar como texto, as linhas serão `bytes`, não `str`. Converta com `.decode()` ou abra em modo texto.
4. **Não especificar `encoding`** – Em ambientes onde o padrão não é UTF‑8, caracteres podem ser corrompidos. Sempre indique `encoding='utf-8'` (ou o apropriado).

### Dicas de performance

- **Ajuste do buffer:** `open(..., buffering=65536)` aumenta o tamanho do buffer (64 KB) e pode melhorar a taxa de leitura em discos rápidos.
- **Desativar a conversão de nova linha:** `open(..., newline='')` evita a tradução automática de `\r\n` para `\n` quando isso não for necessário.
- **Processamento em lote:** Se precisar de operações custosas por linha, agrupe-as em blocos (`chunks`) usando `itertools.islice` para reduzir a sobrecarga de chamadas de função.

---

### Resumindo

A forma mais “Pythonic” e eficiente de ler um arquivo linha a linha é:

```python
with open("caminho/arquivo.txt", "r", encoding="utf-8") as f:
    for linha in f:
        # processar a linha aqui
        ...
```

Isso combina legibilidade, segurança (fechamento automático) e uso mínimo de memória, funcionando bem para arquivos de qualquer tamanho. Boa leitura!

</details>

## 4. List comprehensions

```bash
pychat ask --raw "O que são list comprehensions? Dê exemplos."
```
<details><summary>Resposta completa (3963 caracteres)</summary>

**Resposta curta:**  
List comprehensions são uma sintaxe concisa para criar listas a partir de iteráveis, combinando laços (`for`) e, opcionalmente, condições (`if`) em uma única expressão.

---

### Explicação detalhada

Em Python, antes das *list comprehensions* era comum usar um `for` tradicional para montar uma lista:

```python
resultado = []
for x in range(10):
    if x % 2 == 0:  # opcional: filtro
        resultado.append(x * x)
```

A mesma lógica pode ser escrita em **uma linha** com a sintaxe de list comprehension:

```python
resultado = [x * x for x in range(10) if x % 2 == 0]
```

A estrutura geral é:

```python
[ <expressão> for <variável> in <iterável> if <condição> ]
```

- **`<expressão>`**: o valor que será colocado na nova lista (pode ser uma chamada de função, uma tupla, etc.).
- **`for <variável> in <iterável>`**: o laço que itera sobre o iterável original.
- **`if <condição>`** (opcional): filtro que decide se o elemento será incluído.

#### Por que usar list comprehensions?

1. **Legibilidade** – Quando a lógica é simples, a expressão fica mais curta e fácil de ler.
2. **Performance** – Internamente, a list comprehension é otimizada em C; costuma ser mais rápida que um `for` explícito que usa `append`.
3. **Imutabilidade** – Como a lista é criada de uma vez, reduz a chance de erros de estado mutável.

#### Exemplos práticos

```python
# 1. Quadrados dos números de 0 a 9
quadrados = [x**2 for x in range(10)]
print(quadrados)
# Saída: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 2. Apenas números pares ao quadrado
pares_quadrados = [x**2 for x in range(10) if x % 2 == 0]
print(pares_quadrados)
# Saída: [0, 4, 16, 36, 64]

# 3. Transformar strings em maiúsculas
palavras = ["python", "list", "comprehension"]
maiúsculas = [p.upper() for p in palavras]
print(maiúsculas)
# Saída: ['PYTHON', 'LIST', 'COMPREHENSION']

# 4. Criar pares (tupla) a partir de duas listas (produto cartesiano)
a = [1, 2]
b = ["x", "y"]
pares = [(i, j) for i in a for j in b]
print(pares)
# Saída: [(1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')]

# 5. Filtrar dicionário e gerar nova lista de chaves
dados = {"alice": 23, "bob": 30, "carol": 19}
maiores_de_20 = [nome for nome, idade in dados.items() if idade > 20]
print(maiores_de_20)
# Saída: ['alice', 'bob']
```

#### Armadilhas comuns

| Problema | Como evitar |
|----------|--------------|
| **Complexidade excessiva** – muitas cláusulas `for`/`if` tornam a expressão ilegível. | Quando houver mais de 2‑3 níveis, prefira funções auxiliares ou um `for` tradicional. |
| **Modificação de variáveis externas** – usar `list comprehension` para efeitos colaterais (ex.: `print`) não é idiomático. | Use `for` explícito se precisar de efeitos colaterais. |
| **Escopo de variáveis** – a variável de iteração sobrescreve a existente no escopo externo. | Não reutilize nomes de variáveis já usados fora da compreensão. |
| **Memória** – a list comprehension cria a lista inteira na memória. | Se precisar de um iterável “preguiçoso”, use *generator expressions* (`(expr for ...)`). |

#### Alternativas idiomáticas

- **Generator expression** (para consumo sob demanda, não cria lista na memória):

  ```python
  soma = sum(x**2 for x in range(10) if x % 2 == 0)
  ```

- **Funções de ordem superior** (`map`, `filter`) – ainda válidas, mas menos legíveis para a maioria dos casos:

  ```python
  pares_quadrados = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))
  ```

#### Compatibilidade

List comprehensions existem desde o Python 2.0, portanto funcionam em todas as versões atuais, inclusive Python 3.12+ (a versão alvo da resposta). Não há mudanças de sintaxe entre versões.

---

**Resumo:** List comprehensions são uma forma concisa, legível e geralmente mais rápida de gerar listas a partir de iteráveis, combinando laços e filtros em uma única expressão. Use-as quando a lógica for simples; caso contrário, opte por um `for` explícito ou por generator expressions.

</details>

## 5. Tratamento de exceções

```bash
pychat ask --raw "Como tratar exceções com try/except/finally?"
```
<details><summary>Resposta completa (4843 caracteres)</summary>

**Resposta curta:**  
Use `try` para envolver o código que pode gerar exceções, `except` para capturá‑las e tratá‑las, `else` (opcional) para o código que só roda se nada falhar, e `finally` para o que deve ser executado sempre, independentemente do sucesso ou da falha.

---

## Explicação detalhada

| Bloco | Quando é executado | Uso típico |
|-------|--------------------|------------|
| `try` | Sempre que o bloco é iniciado | Coloque aqui o código que pode lançar exceções. |
| `except` | Só se uma exceção for levantada dentro de `try` | Capture tipos específicos (`except ValueError:`) ou todas (`except Exception as e:`). |
| `else` | Só se **nenhuma** exceção for levantada | Código que depende do sucesso do `try`, mas que não deve estar dentro dele. |
| `finally` | Sempre, mesmo que a exceção seja propagada ou o programa termine com `return`/`break`/`continue` | Liberação de recursos (fechar arquivos, conexões, liberar locks, etc.). |

### Por que usar `finally`?

- Garante que recursos sejam liberados mesmo que algo dê errado.
- É executado antes que a exceção seja propagada novamente, permitindo limpeza antes de “sair” do bloco.

### Armadilhas comuns

1. **Capturar exceções genéricas demais** (`except:` ou `except Exception:`) pode esconder bugs inesperados. Prefira capturar tipos específicos.
2. **Esquecer o `else`**: colocar código que não gera exceção dentro de `try` pode mascarar erros de lógica, pois ele será tratado como exceção mesmo que não devesse.
3. **Re‑lançar exceções sem preservação da pilha**: use `raise` sem argumentos dentro do `except` para manter o traceback original.
4. **Usar `finally` para lógica que pode falhar**: se o código dentro de `finally` lançar outra exceção, a original pode ser perdida. Envolva o código de limpeza em outro `try/except` se necessário.

### Exemplo completo (Python 3.12)

```python
def dividir(a, b):
    """Divide a por b e demonstra tratamento de exceções."""
    try:
        # Código que pode gerar exceção (ZeroDivisionError, TypeError)
        resultado = a / b
    except ZeroDivisionError as e:
        # Tratamento específico para divisão por zero
        print(f"Erro: divisão por zero -> {e}")
        # Podemos decidir retornar um valor padrão ou re‑lançar
        return None
    except TypeError as e:
        # Tratamento para tipos incompatíveis (ex.: str / int)
        print(f"Erro: tipos incompatíveis -> {e}")
        return None
    else:
        # Executa somente se nenhum erro ocorreu
        print("Divisão realizada com sucesso.")
        return resultado
    finally:
        # Sempre executado: aqui liberamos recursos fictícios
        print("Finalizando a operação de divisão.\n")


# Testes
print(dividir(10, 2))  # Caso normal
print(dividir(10, 0))  # ZeroDivisionError
print(dividir(10, "a"))  # TypeError
```

#### Saída esperada

```
Divisão realizada com sucesso.
Finalizando a operação de divisão.

5.0
Erro: divisão por zero -> division by zero
Finalizando a operação de divisão.

None
Erro: tipos incompatíveis -> unsupported operand type(s) for /: 'int' and 'str'
Finalizando a operação de divisão.

None
```

### Como funciona o exemplo

1. **`try`** tenta executar `a / b`.  
2. Se `b` for zero, levanta `ZeroDivisionError`, que é capturada pelo primeiro `except`.  
3. Se `b` for de tipo incompatível (ex.: string), levanta `TypeError`, capturada pelo segundo `except`.  
4. Quando não há exceção, o bloco `else` roda, imprimindo a mensagem de sucesso e retornando o resultado.  
5. Independentemente do caminho (sucesso, erro ou retorno antecipado), o bloco `finally` sempre imprime a mensagem de finalização.

### Alternativas idiomáticas

- **Context Managers (`with`)**: para recursos que precisam ser fechados (arquivos, sockets, locks), prefira `with` ao invés de `try/finally`. Ex.:

  ```python
  with open("dados.txt", "r") as f:
      conteudo = f.read()  # O arquivo será fechado automaticamente.
  ```

- **`contextlib.suppress`**: suprime exceções específicas sem precisar de `except` explícito.

  ```python
  from contextlib import suppress

  with suppress(FileNotFoundError):
      os.remove("arquivo_que_não_existe.txt")
  ```

- **`raise from`**: ao re‑lançar uma exceção, preserve a causa original.

  ```python
  try:
      int("abc")
  except ValueError as e:
      raise RuntimeError("Conversão falhou") from e
  ```

### Boas práticas resumidas

1. Capture apenas exceções que você pode tratar de forma útil.  
2. Use `else` para separar o código “normal” do código que pode falhar.  
3. Reserve `finally` para limpeza de recursos críticos.  
4. Quando possível, substitua `try/finally` por *context managers* (`with`).  
5. Sempre preserve o traceback ao re‑lançar (`raise` ou `raise ... from ...`).  

Com esses princípios, seu tratamento de exceções será claro, seguro e fácil de manter. 🚀

</details>

## 6. Pergunta fora do escopo (a recusa é uma regra do system prompt)

```bash
pychat ask --raw "Qual a capital da França?"
```
Resposta:

> Desculpe, mas não posso responder a essa pergunta. Se você tiver alguma dúvida sobre Python ou programação, ficarei feliz em ajudar!

## 7. Memória de conversa (`pychat chat`)

Teste feito com um nome de variável que só o histórico conhece (para o modelo não poder
"adivinhar"):

```text
you> Criei uma lista em Python chamada xpto_compras. Apenas confirme em uma frase.
you> Mostre como adicionar o item pao na lista que acabei de mencionar, usando o nome exato dela.
     -> a resposta usa `xpto_compras`                      (histórico reenviado ao modelo)
you> /reset
Conversation cleared.
you> Mostre como adicionar o item pao na lista que acabei de mencionar, usando o nome exato dela.
     -> a resposta NÃO conhece `xpto_compras`              (memória limpa)
```

## 8. Erros amigáveis (reais)

Modelo inexistente na conta (resposta HTTP 404 do provedor, traduzida):

```text
$ PYCHAT_MODEL=modelo-que-nao-existe pychat ask "oi"
Error: The configured model was not found for this account.
Hint: Set PYCHAT_MODEL to a model you have access to.
```

Sem chave configurada (código de saída `2`):

```text
$ pychat ask "Como criar uma lista em Python?"
Error: OPENAI_API_KEY is not set.
Hint: Export it or put it in a .env file (see .env.example).
```

## 9. O que a execução real revelou

Três defeitos que nenhum teste offline pegaria, todos corrigidos com teste de regressão:

1. **Crash de codificação.** O primeiro `pychat ask` real derrubou o CLI com
   `UnicodeEncodeError`: no Windows, com a saída redirecionada (code page cp1252), um
   caractere que o modelo emite (espaço fino U+202F) não é codificável. Correção:
   `use_utf8_output` reconfigura `stdout`/`stderr` para UTF-8 na inicialização.
2. **Resposta cortada em silêncio.** Com o limite padrão de 1024 tokens, a resposta da
   pergunta 2 terminava no meio de uma frase e nada avisava o usuário (modelos de
   raciocínio ainda gastam parte do limite pensando). Correções: o padrão subiu para
   2048 e a resposta cortada (`finish_reason == "length"`) passa a terminar com um aviso.
   Exemplo real com `PYCHAT_MAX_TOKENS=120`:

```text
... Ela pode conter itens de qualquer

[Answer truncated by the token limit. Raise PYCHAT_MAX_TOKENS.]
```
3. **Streaming duplicado no terminal** (achado pelo autor no PowerShell). O quadro do
   `Live` continha a resposta inteira; quando passava da altura da tela, as linhas que
   rolaram não podiam ser apagadas e cada atualização deixava outra cópia. Correção: durante
   o streaming só a cauda é desenhada (sempre menor que o terminal) e o Markdown completo é
   impresso uma vez no final.
