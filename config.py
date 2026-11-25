"""Configurações compartilhadas por todos os scripts do projeto.

Centraliza caminhos, nomes de colunas e a semente aleatória, para que
preparar.py, explorar.py, treinar.py, prever.py e processar.py usem
exatamente as mesmas definições.
"""
from pathlib import Path

PASTA = Path(__file__).resolve().parent

# Caminhos
DADOS_BRUTOS = PASTA / "dados" / "bruto" / "dataset.csv"
DADOS_LIMPOS = PASTA / "dados" / "processado" / "spotify_limpo.csv"
PASTA_MODELOS = PASTA / "modelos"
ARQUIVO_MODELO = PASTA_MODELOS / "modelo.joblib"
PASTA_RESULTADOS = PASTA / "resultados"
PASTA_FIGURAS = PASTA_RESULTADOS / "figuras"
RELATORIO_PREPARACAO = PASTA_RESULTADOS / "preparacao.json"
ARQUIVO_ORIGEM = PASTA / "dados" / "origem.json"

# Reprodutibilidade (mesma semente usada nas aulas)
SEMENTE = 42

# Variável-alvo e coluna de agrupamento (mesma música => mesmo conjunto)
ALVO = "popularity"
GRUPO = "musica_id"

# Atributos de entrada
NUMERICAS = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duracao_min",
]
BINARIAS = ["explicit", "mode"]
CATEGORICAS = ["key", "time_signature", "track_genre"]
ATRIBUTOS = NUMERICAS + BINARIAS + CATEGORICAS

# Descrição curta de cada atributo (usada na interface e no contexto da LLM)
DESCRICOES = {
    "danceability": "adequação para dança (0 a 1)",
    "energy": "energia/intensidade (0 a 1)",
    "loudness": "volume médio em dB (de -60 a 0; quanto mais perto de 0, mais alto; "
                "no dataset: mediana -7,2 dB, 90% das faixas abaixo de -3,7 dB)",
    "speechiness": "presença de fala (0 a 1)",
    "acousticness": "chance de ser acústica (0 a 1)",
    "instrumentalness": "chance de ser instrumental (0 a 1)",
    "liveness": "presença de plateia/ao vivo (0 a 1)",
    "valence": "positividade/alegria (0 a 1)",
    "tempo": "andamento em BPM (mediana do dataset: 123)",
    "duracao_min": "duração em minutos (mediana do dataset: 3,6)",
    "explicit": "conteúdo explícito (0 = não, 1 = sim)",
    "mode": "modo (0 = menor, 1 = maior)",
    "key": "tonalidade (0 Dó, 1 Dó♯, 2 Ré, 3 Ré♯, 4 Mi, 5 Fá, 6 Fá♯, 7 Sol, "
           "8 Sol♯, 9 Lá, 10 Lá♯, 11 Si)",
    "time_signature": "fórmula de compasso estimada (3 a 7)",
    "track_genre": "gênero musical",
}
