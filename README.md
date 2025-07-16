# Resolução de Sistemas com Métodos Iterativos Usando AWS EC2

Este projeto realiza a resolução de sistemas lineares baseados na matriz de Leontief, utilizando diferentes métodos iterativos implementados em Python. O código foi desenvolvido para ser executado em uma instância **Amazon EC2**, permitindo processamento sequencial dos níveis de agregação (12, 20 e 67 setores).

## ⚙️ Requisitos

- Instância EC2 com Python 3.x
- Acesso à internet (para baixar arquivos do Amazon S3)
- Permissões de leitura pública no bucket S3

## 📂 Arquivos

- `main.py`: Código principal com todos os métodos iterativos e execuções.
- Os arquivos de entrada (`nivel12_A.txt`, `nivel12_f.txt`, etc.) são baixados automaticamente do S3.

## 🧠 Métodos Implementados

- Splitting A (Iteração simples)
- Splitting B (Jacobi)
- Splitting C (Gauss-Seidel)
- Splitting D (SOR)
- Gradiente
- Gradiente Conjugado
- Solução Exata (Eliminação de Gauss para referência)

## 🚀 Execução

```bash
python3 main.py
