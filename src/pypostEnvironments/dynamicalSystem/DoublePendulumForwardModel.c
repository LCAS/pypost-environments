#include "math.h"
#include "Python.h"

#define SQ(x) ((x)*(x))

/*
 * double pendulum forward model
 * 2 modes:
 * mode 1: simulate dt/dst (init 50) iterations -> returns state (phi1, dphi1, phi2, dphi2, phi3, dphi3, phi4, dphi4)
 * mode 2: one iteration (dt==dst) -> returns accelerations (ddphi1, ddphi2, ddphi3, ddphi4)
 *
 **/
typedef enum { false, true } bool;


static PyObject* simulate(PyObject* self, PyObject* args){
    /*matlab calls: xdot = doublePendulum_C_ForwardModel(x, u, dt, m, l, I, g, k, dst);*/
    /*x,u,m,l, I are vectors. all other variables are scalars. */
    /* k denotes the friction coefficient and dst the simulation time step */
    double a1, a1d, a2, a2d;
    double tau1, tau2;  // torques
    double l1, l2; // lengths
    double m1, m2; // masses
    double I1, I2; // Inertia
    double g; // gravity
    double VISCOUS_FRICTION1, VISCOUS_FRICTION2; // Friction
    double dt; // time step
    double dst; // time step simulation
    double pSetPoint1, dSetPoint1, pSetPoint2, dSetPoint2;
    double pGain1, dGain1, pGain2, dGain2;
    int use_pd;

    PyArg_ParseTuple(args, "dddddddddddddddddidddddddd",
                        &a1, &a1d, &a2, &a2d,
                        &tau1, &tau2,
                        &l1, &l2,
                        &m1, &m2,
                        &I1, &I2,
                        &g,
                        &VISCOUS_FRICTION1, &VISCOUS_FRICTION2,
                        &dt,
                        &dst,
                        &use_pd,
                        &pSetPoint1, &dSetPoint1, &pSetPoint2, &dSetPoint2,
                        &pGain1, &dGain1, &pGain2, &dGain2);

   // double ;
  /*  xValues = mxGetPr(prhs[0]);
    u_t = mxGetPr(prhs[1]);
    dt_t = mxGetPr(prhs[2]);
    m_t = mxGetPr(prhs[3]);
    l_t = mxGetPr(prhs[4]);
    I_t = mxGetPr(prhs[5]);
    g_t = mxGetPr(prhs[6]);
    k_t = mxGetPr(prhs[7]);
    dst_t = mxGetPr(prhs[8]);
    PDSetPoints = mxGetPr(prhs[9]);
    PDGains = mxGetPr(prhs[10]);
*/

    /*printf("%f, %f %f, %f %f, %f %f, %f\n", dt, m1, m2, l1, l2, I1, I2, dst);*/

    int numIts = (int)round(dt/dst);

    bool flgOnlyAccs = false;
    if(dt-1e-3 < 1.0 && dt+1e-3 > 1.0)
        flgOnlyAccs = true;


//    if(flgOnlyAccs)
//       plhs[0] = mxCreateDoubleMatrix(2, 1, mxREAL); /* MODE 2 */
//    else
//        plhs[0] = mxCreateDoubleMatrix(6, 1, mxREAL); /* MODE 1 */

//   double *outArray;
//   outArray = mxGetPr(plhs[0]);

    a1 += M_PI;

    double s1, c1, s2, c2;
  	double h11, h12, h21, h22, b1, b2;
  	double determinant;
  	double a1dd, a2dd;

	double l1CM = l1 / 2.0;
	double l2CM = l2 / 2.0;

    double tmpTau1, tmpTau2, t1, t2;


    int tIndex;
    for(tIndex=0;tIndex<numIts;tIndex++){

        s1 = sin( a1 );
        c1 = cos( a1 );
        s2 = sin( a2 );
        c2 = cos( a2 );

        /*printf("a1 %f s1 %f c1 %f\n",a1 , s1, c1);
        printf("a2 %f s2 %f c2 %f\n",a2 , s2, c2);*/

        h11 = I1 + I2  + l1CM * l1CM * m1 + l1 * l1 * m2 + l2CM * l2CM * m2 + 2 * l1 * l2CM * m2 * c2;
        h12 = I2 + l2CM * l2CM * m2 + l1 * l2CM * m2 * c2;

        b1 = g * l1CM * m1 * s1 + g * l1 * m2 * s1 + g * l2CM * m2 * c2 * s1 - 2 * a1d * a2d * l1 * l2CM * m2 * s2 - a2d * a2d * l1 * l2CM * m2 * s2 + g *l2CM * m2 * c1 * s2;

        h21 = I2 + l2CM * l2CM * m2 + l1 * l2CM * m2 * c2;
        h22 = I2 + l2CM * l2CM * m2;

        b2 = g * l2CM * m2 * c2 * s1 + a1d * a1d * l1 * l2CM * m2 * s2 + g * l2CM * m2 * c1 * s2;

        /*printf("h11 %f h12 %f b1 %f\n",h11 , h12, b1);
        printf("h21 %f h22 %fb2 %f\n",h11 , h12, b2);*/

        /*
         * A = [h11 h12; h21 h22];
         * b = [tau1 - b1 ; tau2 - b2 ];
         * acceleration = A ^-1 * b; %Ainv = [h22 -h12 ; -h21 h11]/determinant;
        */

        tmpTau1 = tau1;
        tmpTau2 = tau2;

        if(use_pd){
            if(a1 < -pSetPoint1){
                tmpTau1 += pGain1 * (-pSetPoint1 - a1) + dGain1 * (-dSetPoint1 - a1d);
            }
            if(a1 >  pSetPoint1){
                tmpTau1 += pGain1 * ( pSetPoint1 - a1) + dGain1 * ( dSetPoint1 - a1d);
            }
            if(a2 < -pSetPoint2){
                tmpTau2 += pGain2 * (-pSetPoint2 -a2) + dGain2 * (-dSetPoint2 - a2d);
            }
            if(a2 >  pSetPoint2){
                tmpTau2 += pGain2 * ( pSetPoint2 - a2) + dGain2 * ( dSetPoint2 - a2d);
            }

           /* printf("PD 1 tau :%f a: %f %f sp: %f %f gains %f %f => tmptau: %f\n",tau1, a1, a1d, PDSetPoints[0], PDSetPoints[1], PDGains[0], PDGains[1], tmpTau1);
            printf("PD 2 tau :%f a: %f %f sp: %f %f gains %f %f => tmptau: %f\n",tau2, a2, a2d, PDSetPoints[2], PDSetPoints[3], PDGains[2], PDGains[3], tmpTau2);*/
        }

        tmpTau1 -= VISCOUS_FRICTION1*a1d;
        tmpTau2 -= VISCOUS_FRICTION2*a2d;

        determinant = h11 * h22 - h12 * h21;

        /*printf("determinant %f \n",determinant);*/

        a1dd = (h22 * (tmpTau1 - b1) - h12 * (tmpTau2 - b2))/determinant;
        a2dd = (h11 * (tmpTau2 - b2) - h21 * (tmpTau1 - b1))/determinant;

        /*printf("a1dd %f a2dd %f\n",a1dd , a2dd);*/

        a1 += dst*a1d;
        a1d += dst*a1dd;
        a2 += dst*a2d;
        a2d += dst*a2dd;
    }

    /*printf("after sim: x1: %f, x2 %f, x3: %f, x4 %f\n", Phi1, dPhi1, Phi2, dPhi2);*/

    if(flgOnlyAccs){
        return Py_BuildValue("dd", a1dd, a2dd);
    }
    else{
        return Py_BuildValue("dddddd", a1 - M_PI, a1d, a2, a2d, t1, t2);
    }

/*
    if(flgOnlyAccs)
    {
        outArray[0] = a1dd;
        outArray[1] = a2dd;
    }
    else
    {
        outArray[0] = (a1 - M_PI);
        outArray[1] = a1d;
        outArray[2] = a2;
        outArray[3] = a2d;
        outArray[4] = t1;
        outArray[5] = t2;
    }
*/
//    return;
}

static PyModuleDef doublePendulumForwardModel_methods[] ={
    {"simulate", simulate, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

/*  define functions in module */
static struct PyModuleDef doublePendulumForwardModel_module =
{
    PyModuleDef_HEAD_INIT,
    "doublePendulumForwardModel", /* name of module */
    NULL,          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    doublePendulumForwardModel_methods,
    NULL
};
/* module initialization */
PyMODINIT_FUNC
PyInit_doublePendulumForwardModel(void){
    PyObject *module = PyModule_Create(&doublePendulumForwardModel_module);
    return module;
}
