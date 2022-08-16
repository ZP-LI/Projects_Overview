/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * averaging_filter.c
 *
 * Code generation for function 'averaging_filter'
 *
 */

/* Include files */
#include "averaging_filter.h"
#include "averaging_filter_data.h"
#include "averaging_filter_emxutil.h"
#include "rt_nonfinite.h"
#include <string.h>

/* Variable Definitions */
static real_T buffer[16];
static emlrtBCInfo emlrtBCI = { -1,    /* iFirst */
  -1,                                  /* iLast */
  16,                                  /* lineNo */
  17,                                  /* colNo */
  "x",                                 /* aName */
  "averaging_filter",                  /* fName */
  "C:\\Users\\LENOVO\\Desktop\\Praktikum MATLAB Simulink for Computer Aided Engineering\\W10\\Tutorial_10_Templates\\Exercise_1\\averaging_"
  "filter.m",                          /* pName */
  0                                    /* checkKind */
};

static emlrtBCInfo b_emlrtBCI = { -1,  /* iFirst */
  -1,                                  /* iLast */
  19,                                  /* lineNo */
  5,                                   /* colNo */
  "y",                                 /* aName */
  "averaging_filter",                  /* fName */
  "C:\\Users\\LENOVO\\Desktop\\Praktikum MATLAB Simulink for Computer Aided Engineering\\W10\\Tutorial_10_Templates\\Exercise_1\\averaging_"
  "filter.m",                          /* pName */
  0                                    /* checkKind */
};

static emlrtRTEInfo emlrtRTEI = { 11,  /* lineNo */
  1,                                   /* colNo */
  "averaging_filter",                  /* fName */
  "C:\\Users\\LENOVO\\Desktop\\Praktikum MATLAB Simulink for Computer Aided Engineering\\W10\\Tutorial_10_Templates\\Exercise_1\\averaging_"
  "filter.m"                           /* pName */
};

/* Function Definitions */
void averaging_filter(const emlrtStack *sp, const emxArray_real_T *x,
                      emxArray_real_T *y)
{
  uint32_T unnamed_idx_1;
  int32_T i;
  int32_T loop_ub;
  int32_T b_i;
  real_T dv[15];
  real_T b_y;

  /*  y = averaging_filter(x) */
  /*  Take an input vector signal 'x' and produce an output vector signal 'y' with */
  /*  same type and shape as 'x' but filtered. */
  /*  Use a persistent variable 'buffer' that represents a sliding window of */
  /*  16 samples at a time. */
  unnamed_idx_1 = (uint32_T)x->size[1];
  i = y->size[0] * y->size[1];
  y->size[0] = 1;
  y->size[1] = (int32_T)unnamed_idx_1;
  emxEnsureCapacity_real_T(sp, y, i, &emlrtRTEI);
  loop_ub = (int32_T)unnamed_idx_1;
  for (i = 0; i < loop_ub; i++) {
    y->data[i] = 0.0;
  }

  i = x->size[1];
  for (b_i = 0; b_i < i; b_i++) {
    /*  Scroll the buffer */
    memcpy(&dv[0], &buffer[0], 15U * sizeof(real_T));
    memcpy(&buffer[1], &dv[0], 15U * sizeof(real_T));

    /*  Add a new sample value to the buffer */
    loop_ub = b_i + 1;
    if ((loop_ub < 1) || (loop_ub > x->size[1])) {
      emlrtDynamicBoundsCheckR2012b(loop_ub, 1, x->size[1], &emlrtBCI, sp);
    }

    b_y = x->data[loop_ub - 1];
    buffer[0] = b_y;

    /*  Compute the current average value of the window and */
    /*  write result */
    for (loop_ub = 0; loop_ub < 15; loop_ub++) {
      b_y += buffer[loop_ub + 1];
    }

    loop_ub = b_i + 1;
    if ((loop_ub < 1) || (loop_ub > y->size[1])) {
      emlrtDynamicBoundsCheckR2012b(loop_ub, 1, y->size[1], &b_emlrtBCI, sp);
    }

    y->data[loop_ub - 1] = b_y / 16.0;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b(sp);
    }
  }
}

void averaging_filter_init(void)
{
  memset(&buffer[0], 0, 16U * sizeof(real_T));
}

/* End of code generation (averaging_filter.c) */
