import urllib.request
import time
import os

# === Configurações do S3 ===
BUCKET_NAME = "bucketdofilipe"
ARQUIVOS = [
    "nivel12_A.txt", "nivel12_f.txt",
    "nivel20_A.txt", "nivel20_f.txt",
    "nivel67_A.txt", "nivel67_f.txt"
]

# === Download dos arquivos direto do S3 sem credenciais ===
def baixar_arquivos_do_s3():
    for arquivo in ARQUIVOS:
        url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{arquivo}"
        print(f"Baixando {arquivo} de {url} ...")
        try:
            urllib.request.urlretrieve(url, arquivo)
            print(f"Arquivo {arquivo} baixado com sucesso.")
        except Exception as e:
            print(f"Erro ao baixar {arquivo}: {e}")

# === Funções utilitárias ===
def ler_matriz(nome_arquivo):
    matriz = []
    with open(nome_arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            matriz.append(list(map(float, linha.strip().split())))
    return matriz

def ler_vetor(nome_arquivo):
    vetor = []
    with open(nome_arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            vetor.append(float(linha.strip()))
    return vetor

def identidade(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def subtrair_matrizes(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def subtrair_vetores(a, b):
    return [a[i] - b[i] for i in range(len(a))]

def multiplicar_matriz_vetor(A, x):
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]

def norma_inf(x1, x2):
    return max(abs(a - b) for a, b in zip(x1, x2))

# === Métodos Iterativos ===
def splitting_A(A, f, max_iter=100):
    x = [0.0] * len(f)
    start = time.time()
    for _ in range(max_iter):
        x = [sum(A[i][j] * x[j] for j in range(len(f))) + f[i] for i in range(len(f))]
    end = time.time()
    return x, max_iter, end - start

def splitting_B(B, f, max_iter=100, tol=1e-6):
    n = len(f)
    x = [0.0] * n
    D = [B[i][i] for i in range(n)]
    R = [[(0 if i == j else B[i][j]) for j in range(n)] for i in range(n)]

    start = time.time()
    for it in range(max_iter):
        x_new = [(f[i] - sum(R[i][j] * x[j] for j in range(n))) / D[i] for i in range(n)]
        if norma_inf(x, x_new) < tol:
            return x_new, it+1, time.time() - start
        x = x_new
    return x, max_iter, time.time() - start

def splitting_C(B, f, max_iter=100, tol=1e-6):
    n = len(f)
    x = [0.0] * n
    start = time.time()
    for it in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(B[i][j] * x_new[j] for j in range(i))
            s2 = sum(B[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (f[i] - s1 - s2) / B[i][i]
        if norma_inf(x, x_new) < tol:
            return x_new, it+1, time.time() - start
        x = x_new
    return x, max_iter, time.time() - start

def splitting_D(B, f, w=1.1, max_iter=100, tol=1e-6):
    n = len(f)
    x = [0.0] * n
    start = time.time()
    for it in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(B[i][j] * x_new[j] for j in range(i))
            s2 = sum(B[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (1 - w) * x[i] + w * (f[i] - s1 - s2) / B[i][i]
        if norma_inf(x, x_new) < tol:
            return x_new, it+1, time.time() - start
        x = x_new
    return x, max_iter, time.time() - start

def gradiente(B, f, max_iter=1000, tol=1e-6):
    n = len(f)
    x = [0.0] * n
    r = subtrair_vetores(f, multiplicar_matriz_vetor(B, x))
    start = time.time()
    for it in range(max_iter):
        Ar = multiplicar_matriz_vetor(B, r)
        rr = sum(r[i]*r[i] for i in range(n))
        alpha = rr / sum(r[i]*Ar[i] for i in range(n))
        x_new = [x[i] + alpha * r[i] for i in range(n)]
        r_new = [r[i] - alpha * Ar[i] for i in range(n)]
        if norma_inf(x, x_new) < tol:
            return x_new, it+1, time.time() - start
        x = x_new
        r = r_new
    return x, max_iter, time.time() - start

def gradiente_conjugado(B, f, max_iter=1000, tol=1e-6):
    n = len(f)
    x = [0.0] * n
    r = subtrair_vetores(f, multiplicar_matriz_vetor(B, x))
    p = r[:]
    start = time.time()
    for it in range(max_iter):
        Ap = multiplicar_matriz_vetor(B, p)
        alpha = sum(r[i]*r[i] for i in range(n)) / sum(p[i]*Ap[i] for i in range(n))
        x_new = [x[i] + alpha*p[i] for i in range(n)]
        r_new = [r[i] - alpha*Ap[i] for i in range(n)]
        if norma_inf(r, r_new) < tol:
            return x_new, it+1, time.time() - start
        beta = sum(r_new[i]*r_new[i] for i in range(n)) / sum(r[i]*r[i] for i in range(n))
        p = [r_new[i] + beta*p[i] for i in range(n)]
        x = x_new
        r = r_new
    return x, max_iter, time.time() - start

# === Solver direto via Eliminação de Gauss ===
def gauss(A, b):
    n = len(A)
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(A[r][i]))
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]
        for j in range(i+1, n):
            f = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= f * A[i][k]
            b[j] -= f * b[i]
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i+1, n))) / A[i][i]
    return x

# === Função principal para resolver cada nível ===
def resolver_nivel(nome_base):
    print(f"\n=== RESULTADOS PARA {nome_base.upper()} ===")
    A = ler_matriz(f"{nome_base}_A.txt")
    f = ler_vetor(f"{nome_base}_f.txt")

    n = len(f)
    I = identidade(n)
    B = subtrair_matrizes(I, A)

    x_exact = gauss([linha[:] for linha in B], f[:])

    x_A, it_A, t_A = splitting_A(A, f)
    x_B, it_B, t_B = splitting_B(B, f)
    x_C, it_C, t_C = splitting_C(B, f)
    x_D, it_D, t_D = splitting_D(B, f)
    x_G, it_G, t_G = gradiente(B, f)
    x_CG, it_CG, t_CG = gradiente_conjugado(B, f)

    for i in range(n):
        print(f"Setor {i+1:02d} | Demanda: {f[i]:,.2f} | A: {x_A[i]:,.2f} | Jacobi: {x_B[i]:,.2f} | GS: {x_C[i]:,.2f} | SOR: {x_D[i]:,.2f} | Grad: {x_G[i]:,.2f} | CG: {x_CG[i]:,.2f} | Exata: {x_exact[i]:,.2f}")

    print("\n--- RESUMO DOS MÉTODOS ---")
    print(f"Splitting A: Iterações = {it_A}, Tempo = {t_A:.4f} s")
    print(f"Jacobi (B): Iterações = {it_B}, Tempo = {t_B:.4f} s")
    print(f"Gauss-Seidel (C): Iterações = {it_C}, Tempo = {t_C:.4f} s")
    print(f"SOR (D): Iterações = {it_D}, Tempo = {t_D:.4f} s")
    print(f"Gradiente: Iterações = {it_G}, Tempo = {t_G:.4f} s")
    print(f"Gradiente Conjugado: Iterações = {it_CG}, Tempo = {t_CG:.4f} s")

# === Execução principal ===
if __name__ == "__main__":
    baixar_arquivos_do_s3()
    resolver_nivel("nivel12")
    resolver_nivel("nivel20")
    resolver_nivel("nivel67")