/*
 * enzpy - Extended Nijboer-Zernike implementation for Python
 * Copyright 2016-2018 J. Antonello <jacopo@antonello.org>
 *
 * This file is part of enzpy.
 *
 * enzpy is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * enzpy is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with enzpy.  If not, see <http://www.gnu.org/licenses/>.
 */


#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <assert.h>
#include <pthread.h>

#define VNMPOCNP_MAX_WORKERS 16

/* DEBUG */
#undef VNMPOCNP_DEBUG_BOUNDS
#undef VNMPOCNP_DEBUG_FORMULA
#undef VNMPOCNP_DEBUG_PTHREAD

/* alternative besseljs, may not compile */
#undef VNMPOCNP_USE_GSL
#undef VNMPOCNP_USE_AMOS

/* broken / unfinished */
#undef VNMPOCNP_SUB_SINGULARITY

/* TODO */
#undef USE_SIGINT

#ifdef VNMPOCNP_USE_GSL
#include <gsl/gsl_sf_bessel.h>
#endif
#ifdef VNMPOCNP_USE_AMOS
/*
 * Including bessel function from AMOS.
 * http://difdop.polytechnique.fr/wiki/index.php/How_to_Bessel_Functions_in_C
 */
extern void zbesj_(double*, double*, double*, int*,
		   int*, double*, double*, int*, int*);

double amos_besselj(int nui, double x)
{
	int kode = 1;
	int n = 1;
	double zr = x;
	double zi = 0.0;
	int nz, ierr;
	double cyr[1], cyi[1];
	double nud = nui;

	zbesj_(&zr, &zi, &nud, &kode, &n, cyr, cyi, &nz, &ierr);

	if (ierr != 0)
		printf("error in amos_besselj\n");

	return cyr[0];
}
#endif


static void *launch(void *p);


typedef struct {
	double complex *outc;

	size_t slicesize;

	int64_t *nvec;
	int64_t *mvec;
	size_t Nb;

	double (*bf)(int, double);

	double *rpr;
	size_t nr;

	double complex *fpc;
	size_t nf;

	size_t pending;
	pthread_mutex_t mux;
	/* pthread_cond_t cond; */
	pthread_t *workers;
	size_t nworkers;
	int stop;

	int verb;

	size_t L_max;
} work_t;


typedef struct {
	int id;
	work_t *w;
} worker_arg_t;


static inline int check_cancel(worker_arg_t *wa)
{
	/* TODO
	 * http://stackoverflow.com/questions/14707049
	 * http://bytes.com/topic/python/answers/168365
	 * -keyboardinterrupt-vs-extension-written-c
	 * http://stackoverflow.com/questions/14707049/
	 */
#ifdef USE_SIGINT
	if (somethinghere()) {
		if (wa) {
			pthread_mutex_lock(&wa->w->mux);
			wa->w->stop = 1;
			pthread_mutex_unlock(&wa->w->mux);
#ifdef VNMPOCNP_DEBUG_PTHREAD
			printf("vnmpocnp: worker %d detected stop\n", wa->id);
#endif
		}
		return 1;
	} else
		return 0;
#else
	return 0;
#endif
}


static double nchoosek(int n, int k)
{
	unsigned long long int r = 0;
	int i;

	if (k < 0 || k > n)
		return 0;
	if (k > n - k)
		k = n - k;

	for (r = 1, i = 1; i <= k; i++) {
		r = r * (n - (k - i));
		r = r / i;
	}

	return (double)r;
}


static inline size_t size_vlj(size_t L, size_t p)
{
	return sizeof(double) * L * (p + 1);
}


static inline int index_vlj(size_t l, size_t L, size_t j, size_t p)
{
#ifdef VNMPOCNP_DEBUG_BOUNDS
	int i;
	i = L * j + (l - 1);
	if (i < (size_vlj(L, p) / sizeof(double)))
		return i;
	else {
		printf("error in indexing vlj\n");
		return 0;
	}
#else
	return L * j + (l - 1);
#endif
}


static double *make_vlj(size_t L, size_t n, size_t m)
{
	double *dp;
	size_t l, j, p, q;
	double min1p;

	q = (n + m) / 2;
	p = (n - m) / 2;

	dp = malloc(size_vlj(L, p));
	if (!dp)
		return NULL;

	min1p = pow(-1, p);
	for (l = 1; l <= L; l++)
		for (j = 0; j <= p; j++) {
			dp[index_vlj(l, L, j, p)] = min1p *
						    (m + l + 2 * j) *
						    ((nchoosek(m + j + l - 1, l - 1) *
						      nchoosek(j + l - 1, l - 1) *
						      nchoosek(l - 1, p - j)) /
						     nchoosek(q + l + j, l));
#ifdef VNMPOCNP_DEBUG_FORMULA
			printf("vnmpocnp: *** vlj[%zu, %zu] = %g\n",
			       l, j,
			       dp[index_vlj(l, L, j, p)]);
#endif
		}

	return dp;
}


static double insum(size_t n, size_t m, size_t l,
		    size_t L, double r, double *vljmap,
		    double (*bf)(int, double))
{
	size_t j, p;
	double vlj, s;

#ifdef VNMPOCNP_SUB_SINGULARITY
	size_t nhalf = n / 2;
#endif

	p = (n - m) / 2;

	s = 0.0;

#ifdef VNMPOCNP_SUB_SINGULARITY
	if (r == 0.0) {
		if (m == 0 && (l - 1) >= nhalf) {
			s = pow(0.5, l) * (nchoosek(l - 1, nhalf) /
					   nchoosek(nhalf + l, l));
			if (nhalf % 2)
				return -s;
			else
				return s;
		} else
			return 0.0;
	}
#endif

	for (j = 0; j <= p; j++) {
		vlj = vljmap[index_vlj(l, L, j, p)];
		if (vlj != 0.0)  /* (p - j) <= l - 1 */
			s += vlj * bf(m + l + 2 * j, 2 * M_PI * r)
			     / (l * pow(2 * M_PI * r, l));

#ifdef VNMPOCNP_DEBUG_BOUNDS
		if (!((p - j) <= (l - 1)) && vlj != 0.0)
			printf("!((p - j) <= (l - 1)) && vlj != 0.0\n");
#endif
	}

#ifdef VNMPOCNP_DEBUG_FORMULA
	printf("vnmpocnp: *** insum = %g\n", s);
#endif

	return s;
}


static inline size_t size_glr(size_t L, size_t nr)
{
	return sizeof(double) * L * nr;
}


static inline int index_glr(size_t l, size_t L, size_t ri, size_t nr)
{
#ifdef VNMPOCNP_DEBUG_BOUNDS
	size_t i;
	i = L * ri + (l - 1);
	if (i < (size_glr(L, nr) / sizeof(double)))
		return i;
	else {
		printf("error in indexing glr\n");
		return 0;
	}
#else
	return L * ri + (l - 1);
#endif
}


double *make_glr(double *vljmap, size_t L, double *r, size_t nr,
		 size_t n, size_t m,
		 double (*bf)(int, double))
{
	double *p;
	size_t l, ri;

	p = malloc(size_glr(L, nr));
	if (!p)
		return NULL;

	for (l = 1; l <= L; l++)
		for (ri = 0; ri < nr; ri++) {
			p[index_glr(l, L, ri, nr)] = insum(n, m, l, L, r[ri], vljmap, bf);
#ifdef VNMPOCNP_DEBUG_FORMULA
			printf("vnmpocnp: *** glj[%zu, %zu] = %g\n",
			       l, ri, p[index_glr(l, L, ri, nr)]);
#endif
		}

	return p;
}


#ifdef VNMPOCNP_DEBUG_BOUNDS
static inline size_t size_out(size_t nr, size_t nf)
{
	return sizeof(double) * nr * nf;
}
#endif


static inline int index_out(size_t ri, size_t nr, size_t fi, size_t nf)
{
#ifdef VNMPOCNP_DEBUG_BOUNDS
	int i;
	i = fi * nr + ri;
	if (i < (size_out(nr, nf) / sizeof(double)))
		return i;
	else {
		printf("error in indexing out\n");
		return 0;
	}
#else
	return fi * nr + ri;
#endif
}


static const char *compute_slice(double complex *outc,
				 size_t n, size_t m,
				 int epsm, double (*bf)(int, double),
				 double *rpr, size_t nr, double complex *fpc, size_t nf,
				 worker_arg_t *wa, size_t L_max)
{
	double *vlj, *glr;
	size_t ri, fi, l;
	double complex tmp, tmpfI, tmp2;

	vlj = make_vlj(L_max, n, m);
	if (!vlj)
		return "make_vlj() failed malloc()";
	glr = make_glr(vlj, L_max, rpr, nr, n, m, bf);
	if (!glr) {
		free(vlj);
		return "make_glr() failed malloc()";
	}

	for (ri = 0; ri < nr; ri++) {
		if (check_cancel(wa)) {
			free(glr);
			free(vlj);
			return "interrupt detected";
		}
		for (fi = 0; fi < nf; fi++) {
			tmpfI = fpc[fi] * I;
			tmp = 0.0 + 0.0 * I;
			for (l = 1; l <= L_max; l++) {
				if ((l - 1) == 0 && creal(tmpfI) == 0.0 && cimag(tmpfI) == 0.0)
					tmp2 = 1.0 + 0.0 * I; /* FIXME skip 0^0, kill the loop in l? */
				else
					tmp2 = cpow(-2.0 * tmpfI, l - 1);
				tmp += tmp2 * glr[index_glr(l, L_max, ri, nr)];
#ifdef VNMPOCNP_DEBUG_FORMULA
				printf("vnmpocnp: slice loop l %zu, "
				       "tmpfI %g%+gi, "
				       "tmp2 %g%+gi, "
				       "tmp %g%+gi\n",
				       l,
				       creal(tmpfI), cimag(tmpfI),
				       creal(tmp2), cimag(tmp2),
				       creal(tmp), cimag(tmp));
#endif
			}
			outc[index_out(ri, nr, fi, nf)] = epsm * cexp(tmpfI) * tmp;
#ifdef VNMPOCNP_DEBUG_FORMULA
			printf("vnmpocnp: *** slice outc %g%+gi\n",
			       creal(outc[index_out(ri, nr, fi, nf)]),
			       cimag(outc[index_out(ri, nr, fi, nf)]));
#endif
		}
	}

	free(vlj);
	free(glr);

	return NULL;
}


static inline int epsm(int mm)
{
	if ((mm < 0) && (abs(mm) % 2) == 1)
		return -1;
	else
		return 1;
}


static inline void printout(size_t i, work_t *w)
{
	if (w->verb == 1)
		printf("vnmpocnp: %.3lf %zu/%zu\n",
		       ((double)i) / ((double)w->Nb), i, w->Nb);
}


static void *launch(void *p)
{
	const char *fail;

#ifdef VNMPOCNP_DEBUG_PTHREAD
	int id;
#endif
	size_t slice;
	work_t *w;
	worker_arg_t *wa = (worker_arg_t*)p;

#ifdef VNMPOCNP_DEBUG_PTHREAD
	id = wa->id;
#endif
	w = wa->w;

#ifdef VNMPOCNP_DEBUG_PTHREAD
	printf("vnmpocnp: worker %d start\n", id);
#endif

	while (1) {
		pthread_mutex_lock(&w->mux);
		if (w->stop) {
#ifdef VNMPOCNP_DEBUG_PTHREAD
			printf("vnmpocnp: worker %d stop\n", id);
#endif
			pthread_mutex_unlock(&w->mux);
			pthread_exit(NULL);
		}
		slice = w->pending;
		if (slice >= 1) {
#ifdef VNMPOCNP_DEBUG_PTHREAD
			printf("vnmpocnp: worker %d slice %zu/%zu\n", id, slice, w->Nb);
#endif
			printout(slice, w);
			w->pending--;
			slice--;
		} else {
#ifdef VNMPOCNP_DEBUG_PTHREAD
			printf("vnmpocnp: worker %d finish\n", id);
#endif
			pthread_mutex_unlock(&w->mux);
			pthread_exit(NULL);
		}
		pthread_mutex_unlock(&w->mux);

		fail = compute_slice(
			w->outc + w->slicesize * slice,
			w->nvec[slice], abs(w->mvec[slice]), epsm(w->mvec[slice]),
			w->bf,
			w->rpr, w->nr,
			w->fpc, w->nf,
			wa,
			w->L_max);
		if (fail) {
			pthread_mutex_lock(&w->mux);
			w->stop = 1;
			pthread_mutex_unlock(&w->mux);

#ifdef VNMPOCNP_DEBUG_PTHREAD
			printf("vnmpocnp: worker %d fail slice %zu/%zu\n", id, slice, w->Nb);
#endif
			pthread_exit(NULL);
		}
	}
}


/*
 * r_n:           number of nodes in r
 * r_dp:          vector of doubles of dimension r_n
 * f_n:           number of nodes in f
 * f_cp:          vector of complex numbers of dimension f_n
 * beta_n:        number of Zernike modes
 * n_ip:          vector of integers of dimension beta_n
 * m_ip:          vector of integers of dimension beta_n
 *
 * L_max:         truncation of summation, default is 35
 * bessel_name:   name of bessel function provider
 * workers_num:   number of pthreads
 * verb:
 *
 * vnm_cp:        third-party allocated array of r_n x f_n x beta_n
 *                complex numbers
 *
 * error:         error message
 *
 * returns 0 on success and sets error to NULL else returns
 * non zero and sets error.
 *
 * Computes
 * \begin{equation}
 *    V_{n}^{m}(r, f) =
 *    \epsilon_m \exp(if)\sum_{l=1}^{L_\text{max}}
 *        (-2if)^{l - 1}
 *        \sum_{j=0}^{(n - |m|)/2} v_{l,j}
 *    \frac{J_{|m| + l + 2j}(2\pi r)}{l(2\pi r)^l},
 * \end{equation}
 * \begin{equation}
 * where $r$ is the normalised radial coordinate of the pupil and $r$ is
 * normalised to $\Lambda/s_0$ and $s_0 = \sin(\alpha_\text{max}) = a/R$.
 * $a$ is the exit radius of the pupil and $R$ is the radius of the exit
 * pupil sphere.
 *
 * See eq. (2.48) in [Braat2008].
 *
 * [Braat2008] J. Braat, S. van Haver, A. Janssen, P. Dirksen, Chapter 6
 * Assessment of optical systems by means of point-spread functions,
 * In: E. Wolf, Editor(s), Progress in Optics, Elsevier, 2008, Volume 51,
 * Pages 349-468, ISSN 0079-6638, ISBN 9780444532114.
 * doi: 10.1016/S0079-6638(07)51006-1
 */


int vnmpocnp(
	size_t r_n, double *r_dp,
	size_t f_n, double complex *f_cp,
	size_t beta_n, int64_t *n_ip, int64_t *m_ip,
	int L_max2,
	const char *bessel_name,
	int workers_num,
	int verb,
	double complex *vnm_cp,
	const char **error)
{
	int retval;
	char *fail2;
	size_t i;
	size_t L_max;
	void *vd;

	work_t work;
	worker_arg_t *worker_args;

	retval = 0;
	memset(&work, 0, sizeof(work_t));

	if (L_max2 <= 0)
		L_max = 35;
	else
		L_max = abs(L_max2);
	if (r_n == 0 || f_n == 0 || beta_n == 0)
		return 0;

	work.nr = r_n;
	work.nf = f_n;
	work.rpr = r_dp;
	work.fpc = f_cp;
	work.L_max = L_max;

	/* bessel provider */
	work.bf = jn;
	if (bessel_name) {
		if (strcmp(bessel_name, "jn") == 0)
			work.bf = jn;
#ifdef VNMPOCNP_USE_GSL
		else if (strcmp(bessel_name, "gsl") == 0)
			work.bf = gsl_sf_bessel_Jn;
#endif
#ifdef VNMPOCNP_USE_AMOS
		else if (strcmp(bessel_name, "amos") == 0)
			work.bf = amos_besselj;
#endif
		else {
			*error = "bessel_name: must be jn, gsl or amos";
			return -1;
		}
	}

	work.verb = verb;

	/* workers */
	work.nworkers = workers_num;
	if (work.nworkers <= 0 || work.nworkers > VNMPOCNP_MAX_WORKERS) {
		*error = "workers_num: illegal value";
		return -1;
	}
	work.workers = malloc(sizeof(pthread_t) * work.nworkers);
	if (!work.workers) {
		*error = "failed malloc for workers";
		retval = -1;
		goto fail_workers_allocate;
	}
	worker_args = malloc(sizeof(worker_arg_t) * work.nworkers);
	if (!worker_args) {
		*error = "failed malloc for worker_args";
		retval = -1;
		goto fail_worker_args;
	}

	/* n,m */
	work.Nb = beta_n;
	if (work.nworkers > work.Nb) {
		*error = "workers_num > beta_n, only parallel computation for n and m"
			 " is implemented";
		retval = -1;
		goto fail_worker_args;
	}
	work.nvec = n_ip;
	work.mvec = m_ip;
	for (i = 0; i < work.Nb; i++) {
		if (!((n_ip[i] - abs(m_ip[i])) >= 0) ||
		    !(fmod(n_ip[i] - abs(m_ip[i]), 2) == 0.0)) {
			*error = "n - |m| must be a non negative integer and even";
			goto fail_worker_args;
		}
	}

	/* create output of dimension nr x nf x Nb */
	/*
	   work.outc = malloc(sizeof(double complex)*work.nr*work.nf*work.Nb);
	 */
	work.outc = vnm_cp;
	work.slicesize = work.nr * work.nf;

	fail2 = NULL;
	if (work.nworkers == 1) {
		for (i = 0; i < work.Nb; i++) {
			*error = compute_slice(
				work.outc + work.slicesize * i,
				work.nvec[i], abs(work.mvec[i]), epsm(work.mvec[i]),
				work.bf,
				work.rpr, work.nr,
				work.fpc, work.nf,
				NULL,
				work.L_max);
			if (*error) {
				retval = -1;
				goto fail_exit;
			}
			printout(i, &work);
		}
	} else {
		pthread_mutex_init(&work.mux, NULL);
		/* pthread_cond_init(&work.cond); */
		work.pending = work.Nb;

		for (i = 0; i < work.nworkers; i++) {
			worker_args[i].id = i;
			worker_args[i].w = &work;
#ifdef VNMPOCNP_DEBUG_PTHREAD
			printf("vnmpocnp: start thread %zu/%zu\n", i, work.nworkers);
#endif
			pthread_create(work.workers + i, NULL, &launch,
				       (void*)(worker_args + i));
		}

		for (i = 0; i < work.nworkers; i++) {
#ifdef VNMPOCNP_DEBUG_PTHREAD
			printf("vnmpocnp: wait for thread %zu/%zu\n", i, work.nworkers);
#endif
			vd = (void*)fail2;
			pthread_join(work.workers[i], &vd);
			if (fail2 && !error) {
#ifdef VNMPOCNP_DEBUG_PTHREAD
				printf("vnmpocnp: fail %s for thread %zu/%zu\n", fail2,
				       i, work.nworkers);
#endif
				*error = fail2;
				retval = -1;
			}
		}

#ifdef VNMPOCNP_DEBUG_PTHREAD
		printf("vnmpocnp: finished %zu threads\n", work.nworkers);
#endif
		/*
		         pthread_mutex_lock(work.mux);
		   while (work.pending)
		        pthread_cond_wait(work.cond, work.mux);
		   pthread_mutex_unlock(work.mux);
		 */
	}

fail_exit:
fail_worker_args:
	free(worker_args);
fail_workers_allocate:
	free(work.workers);

	return retval;
}
