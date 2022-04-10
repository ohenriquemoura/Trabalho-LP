// Código para calcular matrizes simétircas A x B = C 
#include <iostream>
#include <stdint.h>
const uint16_t n = 32; // todas as matrizes teram tamanho n x n 
                       // todas as matrizes serão de tamanho n x n
                      //este número pode ser aumentado obviamente para números muito grandes talvez desligue a saída do terminal no final do programa
// Cada thread calcula um valor da matriz de resultados
// Calcula  A x B = C
global void mat_mul_kn(float *p_C, const float *p_A, const float *p_B)
{
    const uint16_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint16_t j = threadIdx.y + blockIdx.y * blockDim.y;
 
    if(i >= n || j >= n) return;
 
    // indice na matriz de resultados
    const uint32_t ij = i + j * n;
 
    // calculo real
    float sum = 0.0; //
variável local para evitar acessar a memória global mais do que precisamos
 
    for(uint16_t l = 0; l < n; ++l)
        sum += p_A[l + j * n] * p_B[i + l * n];
 
    // salva o resultado (acessa a memória global apenas uma vez)
    p_C[ij] = sum;
}
 
// Cada thread calcula um valor da matriz resultante usando memória compartilhada
// Computes  A x B = C
global void mat_mul_sh_kn(float *p_C, const float *p_A, const float *p_B)
{
    const uint16_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const uint16_t j = threadIdx.y + blockIdx.y * blockDim.y;
 
    if(i >= n || j >= n) return;
    // get shared memory in one continued array
    shared float s_A[n * n];
    shared float s_B[n * n];
 
    // índice na matriz resultante
    const uint32_t ij = i + j * n;   // A
    // const uint32_t ji = j + i * n; // B
 
    // Copiar dados da memória global para a memória compartilhada
    s_A[ij] = p_A[ij];   // A
    // s_A[ij] = p_A[ji]; // B
    s_B[ij] = p_B[ij];
 
    __syncthreads();
 
    // O cálculo de fato
    float sum = 0.0; //  Variável local para prevenir o acesso
                     //  a memória local para além dop necessário
 
    for(uint16_t l = 0; l < n; ++l)
        sum += s_A[l + j * n] * s_B[i + l * n];   // A
        // sum += s_A[j + l * n] * s_B[i + l * n]; // B
 
    // salva o resultado (acessa a memória global somente uma vez)
    p_C[ij] = sum;
}
 
// função main do programa
int main(int argc, char *argv[])
{
    int exit = 0;
 
    // ponteiros de dados do host
    float *A, *B, *C;
    // device data pointers
    float *d_A, *d_B, *d_C;
 
    // alocar dados do host
    A = new float[n*n];
    B = new float[n*n];
    C = new float[n*n];
 
    // alocação de dados
    cudaMalloc((void **)&amp;d_A, n*n * sizeof(float));
    cudaMalloc((void **)&amp;d_B, n*n * sizeof(float));
    cudaMalloc((void **)&amp;d_C, n*n * sizeof(float));
 
    // inicializar dados do host
    for(uint16_t i = 0; i < n*n; ++i)
    {
        A[i] = i + 1.0f;
        B[i] = n*n - i;
    }
 
    // copiar e iniciar dados do programa
    cudaMemcpy((void*)d_A, (void*)A, n*n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_B, (void*)B, n*n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset((void*)d_C, 0, n*n * sizeof(float));
 
    // fazer a computação
    dim3 blksz(32,32,1); // block size
    dim3 nblk((n + blksz.x - 1) / blksz.x, (n + blksz.y - 1) / blksz.y, 1); // number of blocks
    mat_mul_kn<<<nblk, blksz>>>(d_C, d_A, d_B);
    mat_mul_sh_kn<<<nblk, blksz>>>(d_C, d_A, d_B);
 
    // buscar e imprimir resultados
    cudaMemcpy((void*)C, (void*)d_C, n*n * sizeof(float), cudaMemcpyDeviceToHost);
    for(uint16_t j = 0; j < n; ++j)
    {
        for(uint16_t i = 0; i < n; ++i)
            std::cout << C[i + j * n] << "\t";
        std::cout << std::endl;
    }
 
    // liberar memória do dispositivo
    cudaFree((void *)d_A);
    cudaFree((void *)d_B);
    cudaFree((void *)d_C);
 
    // liberar memória do host
    delete[] A;
    delete[] B;
    delete[] C;
 
    cudaDeviceReset();
 
    return exit;
}