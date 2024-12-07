/*
 * GMP version of the ranking and unranking method proposed by
 * Kenji Mikawa and Ken Tanaka. 2023. Efficient linear-time ranking and unranking of derangements.
 * Inf. Process. Lett. 179, C (Jan 2023). https://doi.org/10.1016/j.ipl.2022.106288
*/
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ERROR_INVALID_INPUT -1
#define ERROR_NOT_DERANGEMENT -2
#define ERROR_MEMORY_ALLOCATION -3

mpz_t *derangement_count_gmp(int n) {
  mpz_t *derangement = (mpz_t *)malloc(sizeof(mpz_t) * (n + 1));
  if (derangement == NULL) {
    return NULL;
  }
  mpz_init_set_ui(derangement[0], 1);
  mpz_init_set_ui(derangement[1], 0);

  mpz_t temp;
  mpz_init(temp);
  for (int i = 2; i <= n; i++) {
    mpz_init(derangement[i]);

    mpz_add(temp, derangement[i - 1], derangement[i - 2]);
    mpz_mul_ui(temp, temp, i - 1);
    mpz_set(derangement[i], temp);
  }
  mpz_clear(temp);

  return derangement;
}

int check_derangement(int *perm, int n) {
  int *seen = (int *)calloc(n, sizeof(int));
  if (seen == NULL) return ERROR_MEMORY_ALLOCATION;
  for (int i = 0; i < n; i++) {
    if (perm[i] < 0 || perm[i] >= n) {
      free(seen);
      return ERROR_NOT_DERANGEMENT;
    }
    if (seen[perm[i]]) {
      free(seen);
      return ERROR_NOT_DERANGEMENT;
    }
    seen[perm[i]] = 1;
    if (perm[i] == i) {
      free(seen);
      return ERROR_NOT_DERANGEMENT;
    }
  }
  free(seen);
  return 0;
}

/*
 * r: rank of the derangement
 * n: number of elements in the permutation
 * p: returned derangement, uninitialized
 * subf: array of subfactorials, precomputed
 * returns: 0 on success, negative on failure
 */
int unrank_gmp(mpz_t r, int n, int *p, mpz_t *subf) {
  // n must be at least 2
  if (n < 2) return ERROR_INVALID_INPUT;
  // make sure r is within range
  if (mpz_cmp(r, subf[n]) >= 0) return ERROR_INVALID_INPUT;

  mpz_t d1;
  mpz_t d2;
  mpz_t dd;
  mpz_init_set(d1, subf[n - 1]);
  mpz_init_set(d2, subf[n - 2]);
  mpz_init(dd);

  int i, x, y, w;

  int *s = (int *)malloc(sizeof(int) * n);
  if (s == NULL) return ERROR_MEMORY_ALLOCATION;
  for (int i = 0; i < n; i++) {
    s[i] = i;
    p[i] = i;
  }

  while (n >= 2) {
    x = s[--n];
    i = mpz_fdiv_ui(r, n);
    y = s[i];
    // swap p[x] and p[y]
    w = p[x];
    p[x] = p[y];
    p[y] = w;
    mpz_fdiv_q_ui(r, r, n);
    mpz_set(dd, d1);
    if (n >= 2) {
      mpz_set(d1, d2);
      mpz_set(d2, subf[n - 2]);
    }
    if (mpz_cmp(r, dd) >= 0) {
      mpz_sub(r, r, dd);
      // swap s[n-1] and s[i], remove s[n-1]
      w = s[i];
      s[i] = s[--n];
      s[n] = w;
      if (n >= 2) {
        mpz_set(d1, d2);
        mpz_set(d2, subf[n - 2]);
      }
    }
  }

  mpz_clear(dd);
  free(s);

  return 0;
}

/*
 * n: number of elements in the derangement
 * p: derangement to rank
 * r: returned rank, initialized but not set
 * subf: array of subfactorials, precomputed
 * returns: 0 on success, negative on failure
 */
int rank_gmp(int n, int *p, mpz_t *r, mpz_t *subf) {
  // n must be at least 2
  if (n < 2) return ERROR_INVALID_INPUT;
  // p must be a derangement
  if (check_derangement(p, n) != 0) return ERROR_NOT_DERANGEMENT;

  // allocate auxiliary arrays for computing rank
  int *q = (int *)malloc(sizeof(int) * n);
  int *s = (int *)malloc(sizeof(int) * n);
  int *t = (int *)malloc(sizeof(int) * n);
  int *c_arr = (int *)malloc(sizeof(int) * n);
  int *n_arr = (int *)malloc(sizeof(int) * n);
  mpz_t *d_arr = (mpz_t *)malloc(sizeof(mpz_t) * n);
  if (q == NULL || s == NULL || t == NULL || c_arr == NULL || d_arr == NULL ||
      n_arr == NULL) {
    free(q);
    free(s);
    free(t);
    free(c_arr);
    free(d_arr);
    free(n_arr);
    return ERROR_MEMORY_ALLOCATION;
  }
  // initialize auxiliary arrays
  int idx = 0;
  for (int i = 0; i < n; i++) {
    q[p[i]] = i;
    s[i] = i;
    t[i] = i;
    c_arr[i] = 0;
    n_arr[i] = 0;
    mpz_init_set_ui(d_arr[i], 0);
  }

  int c, w, x, y, z;
  mpz_t d1, d2;
  mpz_init(d1);
  mpz_init(d2);

  while (n > 1) {
    x = s[n - 1];
    y = q[x];
    c = t[p[x]];
    mpz_set(d1, subf[n - 1]);
    mpz_set(d2, subf[n - 2]);

    // swap p[x] and p[y]
    w = p[x];
    p[x] = p[y];
    p[y] = w;
    // swap q[p[x]] and q[p[y]]
    w = q[p[x]];
    q[p[x]] = q[p[y]];
    q[p[y]] = w;

    if (p[y] == y) {
      // swap s[n - 2] and s[t[y]]
      w = s[n - 2];
      s[n - 2] = s[t[y]];
      s[t[y]] = w;
      z = s[t[y]];
      // swap t[z] and t[y]
      w = t[z];
      t[z] = t[y];
      t[y] = w;

      mpz_set(d_arr[idx], d1);
      n_arr[idx] = n - 1;
      c_arr[idx++] = c;
      n -= 2;
    } else {
      // it has been set to 0 in the initialization
      // mpz_set_ui(d_arr[idx], 0);
      n_arr[idx] = n - 1;
      c_arr[idx++] = c;
      n -= 1;
    }
  }

  // compute rank
  mpz_set_ui(*r, 0);
  for (int i = idx - 1; i >= 0; i--) {
    mpz_add(*r, *r, d_arr[i]);
    mpz_mul_si(*r, *r, n_arr[i]);
    mpz_add_ui(*r, *r, (unsigned long)c_arr[i]);
  }

  mpz_clear(d1);
  mpz_clear(d2);
  for (int i = 0; i < n; i++) {
    mpz_clear(d_arr[i]);
  }
  free(q);
  free(s);
  free(t);
  free(c_arr);
  free(d_arr);
  free(n_arr);

  return 0;
}

int derangement_unrank(char *rank, int n, int *perm) {
  mpz_t *subf = derangement_count_gmp(n);
  if (subf == NULL) {
    return ERROR_MEMORY_ALLOCATION;
  }

  mpz_t r;
  mpz_init_set_str(r, rank, 10);
  int status = unrank_gmp(r, n, perm, subf);
  for (int i = 0; i <= n; i++) mpz_clear(subf[i]);
  free(subf);
  mpz_clear(r);

  return status;
}

int derangement_rank(int *perm, int n, char *rank) {
  mpz_t *subf = derangement_count_gmp(n);
  if (subf == NULL) {
    return ERROR_MEMORY_ALLOCATION;
  }

  mpz_t r;
  mpz_init(r);
  int status = rank_gmp(n, perm, &r, subf);
  for (int i = 0; i <= n; i++) mpz_clear(subf[i]);
  free(subf);
  if (status == 0) mpz_get_str(rank, 10, r);
  mpz_clear(r);

  return status;
}
