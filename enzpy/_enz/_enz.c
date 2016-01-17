/*
 * enzpy - Extended Nijboer-Zernike implementation for Python
 * Copyright 2016 J. Antonello <jack@antonello.org>
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


#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <complex.h>


#define __ENZPY__AUTHOR__ "Jacopo Antonello"
#define __ENZPY__COPYRIGHT__ "Copyright 2016, Jacopo Antonello"
#define __ENZPY__LICENSE__ "GPLv3+"
#define __ENZPY__VERSION__ "1.0.0"
#define __ENZPY__EMAIL__ "jack@antonello.org"
#define __ENZPY__STATUS__ "Production"
#define __ENZPY__DOCFORMAT__ "restructuredtext"

extern int vnmpocnp(
	size_t r_n, double *r_dp,
	size_t f_n, double complex *f_cp,
	size_t beta_n, int *n_ip, int *m_ip,
	int L_max,
	const char *bessel_name,
	int workers_num,
	int verb,
	double complex *vnm_cp,
	const char **error);


static inline int not_vect(PyObject *vv)
{
	PyArrayObject *v = (PyArrayObject*)vv;

	if (PyArray_NDIM(v) == 2 &&
	    (PyArray_SIZE(v) == PyArray_DIM(v, 0) ||
	     PyArray_SIZE(v) == PyArray_DIM(v, 1)))
		return 0;
	else if (PyArray_NDIM(v) == 1)
		return 0;
	else
		return 1;
}


static PyObject * enz_vnmpocnp(PyObject *self, PyObject *args,
			       PyObject *keywds)
{
	static char *kwlist[] = { "r",		 "f",		"n",	       "m",	"L_max",
				  "bessel_name", "ncpus",	"verb",	       NULL };
	long syscpus = sysconf(_SC_NPROCESSORS_ONLN);

	PyObject *r_o, *f_o, *n_o, *m_o;
	PyObject *r = NULL;
	PyObject *f = NULL;
	PyObject *n = NULL;
	PyObject *m = NULL;
	const char *error = NULL;
	npy_intp r_n, f_n, beta_n;

	PyObject *vnm = NULL;
	npy_intp vnm_dims[3];

	double *datar = NULL;
	complex double *dataf = NULL;
	int *datan = NULL;
	int *datam = NULL;
	complex double *datap = NULL;

	int kL_max = 35;
	int kncpus = -1;
	int kverb = 0;
	char *kbessel_name = "jn";
	int ret;

	if (syscpus <= 0)
		syscpus = 1;

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!O!|isii",
					 kwlist,
					 &PyArray_Type, &r_o,
					 &PyArray_Type, &f_o,
					 &PyArray_Type, &n_o,
					 &PyArray_Type, &m_o,
					 &kL_max, &kbessel_name, &kncpus, &kverb)) {
		PyErr_SetString(PyExc_SyntaxError, "failed to parse args");
		return NULL;
	}

	r = PyArray_FROM_OTF(r_o, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
	f = PyArray_FROM_OTF(f_o, NPY_COMPLEX128, NPY_ARRAY_IN_ARRAY);
	n = PyArray_FROM_OTF(n_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);
	m = PyArray_FROM_OTF(m_o, NPY_INT64, NPY_ARRAY_IN_ARRAY);

	if (!r) {
		PyErr_SetString(PyExc_ValueError, "cannot convert r to PyArray");
		return NULL;
	}
	if (!f) {
		PyErr_SetString(PyExc_ValueError, "cannot convert f to PyArray");
		return NULL;
	}
	if (!n) {
		PyErr_SetString(PyExc_ValueError, "cannot convert n to PyArray");
		return NULL;
	}
	if (!m) {
		PyErr_SetString(PyExc_ValueError, "cannot convert m to PyArray");
		return NULL;
	}

	if (!r || not_vect(r) || PyArray_TYPE((PyArrayObject*)r) != NPY_FLOAT64) {
		error = "r is not a vector of doubles";
		goto exit_decrement;
	}
	r_n = PyArray_DIM((PyArrayObject*)r, 0);

	if (!f || not_vect(f) ||
	    PyArray_TYPE((PyArrayObject*)f) != NPY_COMPLEX128) {
		error = "f is not a vector of complex numbers";
		goto exit_decrement;
	}
	f_n = PyArray_DIM((PyArrayObject*)f, 0);

	if (!n || not_vect(n) || PyArray_TYPE((PyArrayObject*)n) != NPY_INT64) {
		error = "n is not a vector of integers";
		goto exit_decrement;
	}
	if (!m || not_vect(m) || PyArray_TYPE((PyArrayObject*)m) != NPY_INT64) {
		error = "m is not a vector of integers";
		goto exit_decrement;
	}
	if (PyArray_DIM((PyArrayObject*)n, 0) !=
	    PyArray_DIM((PyArrayObject*)m, 0)) {
		error = "n and m must have the same length";
		goto exit_decrement;
	}
	beta_n = PyArray_DIM((PyArrayObject*)n, 0);

	vnm_dims[0] = r_n;
	vnm_dims[1] = f_n;
	vnm_dims[2] = beta_n;

	vnm = PyArray_New(&PyArray_Type, 3, vnm_dims, NPY_COMPLEX128, NULL, NULL,
			  0, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED, NULL);

	if (!vnm) {
		error = "cannot create vnm";
		goto exit_decrement;
	}

	PyArray_CLEARFLAGS((PyArrayObject*)vnm, NPY_ARRAY_C_CONTIGUOUS);

	assert(PyArray_Size(vnm) == (r_n * f_n * beta_n));
	assert(PyArray_NBYTES(vnm) == (r_n * f_n * beta_n * sizeof(double complex)));
	datap = PyArray_DATA((PyArrayObject*)vnm);

	if (kncpus < 0)
		kncpus = beta_n < syscpus ? beta_n : syscpus;
	else
		kncpus = kncpus > syscpus ? syscpus : kncpus;

	if ((r_n * f_n * beta_n) != 0) {
		datar = PyArray_DATA((PyArrayObject*)r);
		dataf = PyArray_DATA((PyArrayObject*)f);
		datan = PyArray_DATA((PyArrayObject*)n);
		datam = PyArray_DATA((PyArrayObject*)m);
		Py_BEGIN_ALLOW_THREADS
			ret = vnmpocnp(r_n, datar,
				       f_n, dataf,
				       beta_n,
				       datan, datam,
				       kL_max, kbessel_name,
				       kncpus,
				       kverb, datap, &error);
		Py_END_ALLOW_THREADS
		if (ret) {
			Py_XDECREF(vnm);
			goto exit_decrement;
		}
	}

exit_decrement:
	Py_XDECREF(r);
	Py_XDECREF(f);
	Py_XDECREF(n);
	Py_XDECREF(m);
	if (error) {
		PyErr_SetString(PyExc_ValueError, error);
		return NULL;
	} else {
		assert(vnm != NULL);
		return Py_BuildValue("N", vnm);
	}
}


static char vnmpocnp_docstring[] = "vnmpocnp(r, f, n, m, L_max=35, "
				   "bessel_name='fn', ncpus=-1, verb=0).\n"
				   "\n"
				   "Compute\n"
				   "\n"
				   ".. math::\n"
				   "\n"
				   "    V_n^m(r, f) = \\epsilon_m \\exp(if)\n"
				   "        \\sum_{l=1}^{L_max}(-2if)^{l - 1}\n"
				   "        \\sum_{j=0}^{(n - |m|)/2} v_{l,j}\n"
				   "        (1/l(2\\pi r)^l)J_{|m| + l + 2j}(2\\pi r).\n"
				   "\n"
				   "See Eq. (2.48) in [B2008]_.\n"
				   "\n"
				   "Parameters\n"
				   "----------\n"
				   "-   `r`: `numpy` vector of doubles for the radial coordinate\n"
				   "    :math:`r`, which is normalised to the diffraction unit\n"
				   "    `wavelength/NA`\n"
				   "-   `f`: `numpy` vector of complex numbers for the defocus\n"
				   "    parameter, see [J2002]_, [B2008]_, and [H2010]_\n"
				   "-   `n`: `numpy` vector of integers for the radial orders `n`\n"
				   "-   `m`: `numpy` vector of integers for the azimuthal\n"
				   "    frequencies `m`\n"
				   "-   `L_max`: optional `int` for the truncation order of the\n"
				   "    series, see [J2002]_, [B2008]_, and [H2010]_. `L_max <= 0`\n"
				   "    uses the default value of `35`.\n"
				   "-   `bessel_name`: optional `str` to select a Bessel funcion\n"
				   "    provider\n"
				   "-   `ncpus` : optional `int` for the number of threads. `-1`\n"
				   "    chooses all available cpus\n"
				   "-   `verb` : optional `int` to print debugging information\n"
				   "    chooses all available cpus\n"
				   "\n"
				   "Returns\n"
				   "-------\n"
				   "-   `vnm`: `numpy` array of shape `(r.size, f.size, n.size)` \n"
				   "    for :math:`V_n^m(r, f)`\n"
				   "\n"
				   "References\n"
				   "----------\n"
				   "..  [J2002] A. J. E. M. Janssen, \"Extended Nijboer–Zernike\n"
				   "    approach for the computation of optical point-spread\n"
				   "    functions,\" J. Opt. Soc.  Am.  A 19, 849–857 (2002). `doi\n"
				   "    <http://dx.doi.org/10.1364/JOSAA.19.000849>`__.\n"
				   "..  [B2008] J. Braat, S. van Haver, A. Janssen, P. Dirksen,\n"
				   "    Chapter 6 Assessment of optical systems by means of\n"
				   "    point-spread functions, In: E. Wolf, Editor(s), Progress in\n"
				   "    Optics, Elsevier, 2008, Volume 51, Pages 349-468, ISSN\n"
				   "    0079-6638, ISBN 9780444532114. `doi\n"
				   "    <http://dx.doi.org/10.1016/S0079-6638(07)51006-1>`__.\n"
				   "..  [H2010] S. van Haver, The Extended Nijboer-Zernike\n"
				   "    Diffraction Theory and its Applications (Ph.D. thesis,\n"
				   "    Delft University of Technology, The Netherlands, 2010). `doi\n"
				   "    <http://resolver.tudelft.nl/uuid:8d96ba75-24da-4e31-a750-"
				   "1bc348155061>`__.\n";


static PyMethodDef LibenzMethods[] = {
	{ "vnmpocnp", (PyCFunction)enz_vnmpocnp,
	  METH_VARARGS | METH_KEYWORDS, vnmpocnp_docstring },
	{ NULL,	      NULL,			0, NULL }
};


static struct PyModuleDef libenzmodule = {
	PyModuleDef_HEAD_INIT,
	"_enz",
	"Helper library for enzpy.\n\nVersion: " __ENZPY__VERSION__,
	-1,
	LibenzMethods
};


PyMODINIT_FUNC PyInit__enz(void)
{
	PyObject *m = NULL;

	m = PyModule_Create(&libenzmodule);
	if (m == NULL)
		return NULL;

	import_array();

	if (
		PyModule_AddStringConstant(m, "__author__", __ENZPY__AUTHOR__) ||
		PyModule_AddStringConstant(m, "__copyright__", __ENZPY__COPYRIGHT__) ||
		PyModule_AddStringConstant(m, "__license__", __ENZPY__LICENSE__) ||
		PyModule_AddStringConstant(m, "__version__", __ENZPY__VERSION__) ||
		PyModule_AddStringConstant(m, "__email__", __ENZPY__EMAIL__) ||
		PyModule_AddStringConstant(m, "__status__", __ENZPY__STATUS__) ||
		PyModule_AddStringConstant(m, "__copyright__", __ENZPY__COPYRIGHT__) ||
		PyModule_AddStringConstant(m, "__docformat__", __ENZPY__DOCFORMAT__)) {
		Py_DECREF(m);
		return NULL;
	}

	return m;
}
