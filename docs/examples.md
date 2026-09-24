# Exemplos de perguntas e respostas

> **Aviso de transparência.** As respostas abaixo são **ilustrativas**: mostram o formato
> que o prompt do sistema exige (resposta direta, exemplo executável, explicação e
> armadilhas). Não são transcrições de uma execução com GPT-4, porque o repositório não
> embute credenciais. Para gerar respostas reais, rode os comandos com sua `OPENAI_API_KEY`.

As perguntas estão em [`examples/questions.txt`](../examples/questions.txt).
`make demo` executa todas em sequência.

## 1. A pergunta do enunciado

```bash
pychat ask "Como criar uma lista em Python?"
```

Resposta esperada (ilustrativa):

> Em Python, uma lista é criada com colchetes `[]` ou com o construtor `list()`.
>
> ```python
> frutas = ["maçã", "banana", "uva"]   # lista literal
> vazia = []                           # lista vazia
> numeros = list(range(5))             # a partir de um iterável
> print(frutas, vazia, numeros)
> ```
>
> Saída: `['maçã', 'banana', 'uva'] [] [0, 1, 2, 3, 4]`
>
> Listas são **mutáveis** e **ordenadas**: `frutas.append("pera")` adiciona ao final e
> `frutas[0]` acessa o primeiro item. Armadilha comum: `a = b = []` faz `a` e `b`
> apontarem para a *mesma* lista.

## 2. Conversa com memória (`pychat chat`)

```text
you> Como criar uma lista em Python?
pymentor> ... (resposta acima)
you> E como adiciono um item nela?
pymentor> Use append() ...   <- "nela" é resolvido graças ao histórico
you> /reset
Conversation cleared.
```

## 3. Pergunta fora do escopo

```bash
pychat ask "Qual a capital da França?"
```

Comportamento esperado: recusa curta e convite para uma pergunta sobre Python (regra do
system prompt).

## 4. Erros amigáveis (reais, sem chave configurada)

```text
$ pychat ask "Como criar uma lista em Python?"
Error: OPENAI_API_KEY is not set.
Hint: Export it or put it in a .env file (see .env.example).
```

Saída real, verificada nesta base de código; o código de saída é `2`.
