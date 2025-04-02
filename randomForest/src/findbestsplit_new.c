#include <Rmath.h>
#include <R.h>
#include "rf.h"

/* 新函数：绑定抽取组特征 */
void findBestSplit_new(double *x, int *jdex, double *y, int mdim, int nsample,
    int ndstart, int ndend, int *bestVarToReturn, double *decsplit,
    double *bestSplitToReturn, int *ndendl, int *jstat, 
    double sumnode, int nodeCount, int *cat,
    //新增变量
    int huangr_mtry,
    int *huangr_stratum, 
    int *huangr_start, 
    int *huangr_n_vars, 
    int huangr_n_layers
) {
Rprintf("get in findBestSplit_new");
Rprintf("\n");
int last, numCategoriesAllVars[MAX_CAT], icat[MAX_CAT], numCategoriesForVar, nl, nr, npopl, npopr, tieVal;
int tieVar = 1;
int i, j, kv, l, *varIndices, *ncase;
double *xt, *ut, *v, *yl, sumcat[MAX_CAT], avcat[MAX_CAT], tavcat[MAX_CAT], valueAtBestSplit;
double crit, bestSplitForAllVariables, bestSplitWithinVariable, suml, sumr, d, critParent;
//使用临时储存，避免修改原输入参数
int *temp_stratum = (int *) R_Calloc(huangr_n_layers, int);
memcpy(temp_stratum, huangr_stratum, sizeof(int) * huangr_n_layers);
//防止索引溢出，mtry不能大于层数
if (huangr_mtry > huangr_n_layers) {
    error("huangr_mtry exceeds available groups");
}

// 分配内存（与原函数相同部分省略）
ut = (double *) R_Calloc(nsample, double);
xt = (double *) R_Calloc(nsample, double);
v  = (double *) R_Calloc(nsample, double);
yl = (double *) R_Calloc(nsample, double);
varIndices  = (int *) R_Calloc(mdim, int);
ncase = (int *) R_Calloc(nsample, int);
zeroDouble(avcat, MAX_CAT);
zeroDouble(tavcat, MAX_CAT);

/* 绑定组特征选择逻辑 */
int *selected_groups = (int *) R_Calloc(huangr_mtry, int);
int total_selected_vars = 0;
int *group_vars = (int *) R_Calloc(mdim, int); // 存储选中的变量索引
*bestVarToReturn = -1;
*decsplit = 0.0;
bestSplitForAllVariables = 0.0;
valueAtBestSplit = 0.0;

// 随机不放回抽取 huangr_mtry 个组
int last_group = huangr_n_layers - 1;
// 使用临时参数，避免修改原参数
for (i = 0; i < huangr_mtry; ++i) {
  j = (int)(unif_rand() * (last_group + 1));
  selected_groups[i] = temp_stratum[j];
  swapInt(temp_stratum[j], temp_stratum[last_group]);
  last_group--;

  // 调试输出每次交换后的状态
  Rprintf("[SWAP] 交换后临时数组: ");
  for (int g = 0; g < huangr_n_layers; ++g) {
      Rprintf("%d ", temp_stratum[g]);
  }
  Rprintf("\n");
}
// 输出抽取的组索引
Rprintf("Selected groups: ");
for (int g = 0; g < huangr_mtry; ++g) {
    Rprintf("%d ", selected_groups[g]);
}
Rprintf("\n");

// 提取选中组的所有变量
for (i = 0; i < huangr_mtry; ++i) {
    int group_id = selected_groups[i];
    int start_idx = huangr_start[group_id] - 1; // C 索引从 0 开始
    int n_vars = huangr_n_vars[group_id];

    // 错误检查：start_idx和n_vars是否有效
    if (start_idx < 0 || start_idx >= mdim) {
      error("Invalid start index %d for group %d", start_idx, group_id);
  }
  if (n_vars < 0 || (start_idx + n_vars) > mdim) {
      error("Invalid n_vars %d for group %d (start=%d, mdim=%d)",
            n_vars, group_id, start_idx, mdim);
  }
    for (j = 0; j < n_vars; ++j) {
        group_vars[total_selected_vars++] = start_idx + j;
    }
}
//抽取到的所有特征总数不可能大于mdim总特征数
if (total_selected_vars > mdim) {
    error("Selected variables exceed mdim");
}
// 输出抽取的变量总数
Rprintf("Total selected variables: %d\n", total_selected_vars);
// 直接使用抽到的所有特征，而非再次抽取
last = total_selected_vars - 1;
for (i = 0; i < total_selected_vars; ++i) {
    kv = group_vars[i];
    bestSplitWithinVariable = 0.0;
    // 后续分析与原函数相同，使用 kv 作为变量索引
    numCategoriesForVar = cat[kv];
    if (numCategoriesForVar == 1) {
      /* numeric variable */
      for (j = ndstart; j <= ndend; ++j) {
        xt[j] = x[kv + (jdex[j] - 1) * mdim]; /*indexing to represent 2d in a 1d vector */
        yl[j] = y[jdex[j] - 1];
      }
    } else {
      /* categorical variable */
      zeroInt(numCategoriesAllVars, MAX_CAT);
      zeroDouble(sumcat, MAX_CAT);
      for (j = ndstart; j <= ndend; ++j) {
        l = (int) x[kv + (jdex[j] - 1) * mdim];
        sumcat[l - 1] += y[jdex[j] - 1];
        numCategoriesAllVars[l - 1] ++;
      }
      /* Compute means of Y by category. */
      for (j = 0; j < numCategoriesForVar; ++j) {
        avcat[j] = numCategoriesAllVars[j] ? sumcat[j] / numCategoriesAllVars[j] : 0.0;
      }
      /* Make the category mean the `pseudo' X data. */
      for (j = 0; j < nsample; ++j) {
        xt[j] = avcat[(int) x[kv + (jdex[j] - 1) * mdim] - 1];
        yl[j] = y[jdex[j] - 1];
      }
    }
    /* copy the x data in this node. */
    for (j = ndstart; j <= ndend; ++j) {
      v[j] = xt[j];
    }

    for (j = 1; j <= nsample; ++j) {
      ncase[j - 1] = j;
    }

    R_qsort_I(v, ncase, ndstart + 1, ndend + 1);
    
    if (v[ndstart] >= v[ndend]) {
      continue;
    }
    /* ncase(n)=case number of v nth from bottom */
    /* Start from the right and search to the left. */
    critParent = sumnode * sumnode / nodeCount;
    suml = 0.0;
    sumr = sumnode;
    npopl = 0;
    npopr = nodeCount;
    crit = 0.0;
    tieVal = 1;
    /* Search through the "gaps" in the x-variable. */
    for (j = ndstart; j <= ndend - 1; ++j) {
      d = yl[ncase[j] - 1];
      suml += d;
      sumr -= d;
      npopl++;
      npopr--;
      if (v[j] < v[j+1]) {
        crit = (suml * suml / npopl) + (sumr * sumr / npopr) - critParent;
        if (crit > bestSplitWithinVariable) {
          valueAtBestSplit = (v[j] + v[j+1]) / 2.0;
          bestSplitWithinVariable = crit;
          tieVal = 1;
        }
        if (crit == bestSplitWithinVariable) {
          tieVal++;
          if (unif_rand() < 1.0 / tieVal) {
            valueAtBestSplit = (v[j] + v[j+1]) / 2.0;
            bestSplitWithinVariable = crit;
          }
        }
      }
    }
    if (bestSplitWithinVariable > bestSplitForAllVariables) {
      *bestSplitToReturn = valueAtBestSplit;
      *bestVarToReturn = kv + 1;
      bestSplitForAllVariables = bestSplitWithinVariable;
      for (j = ndstart; j <= ndend; ++j) {
        ut[j] = xt[j];
      }
      if (cat[kv] > 1) {
        for (j = 0; j < cat[kv]; ++j) tavcat[j] = avcat[j];
      }
      tieVar = 1;
    }
    if (bestSplitWithinVariable == bestSplitForAllVariables) {
      tieVar++;
      if (unif_rand() < 1.0 / tieVar) {
        *bestSplitToReturn = valueAtBestSplit;
        *bestVarToReturn = kv + 1;
        bestSplitForAllVariables = bestSplitWithinVariable;
        for (j = ndstart; j <= ndend; ++j) {
          ut[j] = xt[j];
        }
        if (cat[kv] > 1) {
          for (j = 0; j < cat[kv]; ++j) tavcat[j] = avcat[j];
        }
      }
    }

}

*decsplit = bestSplitForAllVariables;

  /* If best split can not be found, set to terminal node and return. */
  if (*bestVarToReturn != -1) {
    nl = ndstart;
    for (j = ndstart; j <= ndend; ++j) {
      if (ut[j] <= *bestSplitToReturn) {
        nl++;
        ncase[nl-1] = jdex[j];
      }
    }
    *ndendl = imax2(nl - 1, ndstart);
    nr = *ndendl + 1;
    for (j = ndstart; j <= ndend; ++j) {
      if (ut[j] > *bestSplitToReturn) {
        if (nr >= nsample) break;
        nr++;
        ncase[nr - 1] = jdex[j];
      }
    }
    if (*ndendl >= ndend) *ndendl = ndend - 1;
    for (j = ndstart; j <= ndend; ++j) jdex[j] = ncase[j];

    numCategoriesForVar = cat[*bestVarToReturn - 1];
    if (numCategoriesForVar > 1) {
      for (j = 0; j < numCategoriesForVar; ++j) {
        icat[j] = (tavcat[j] < *bestSplitToReturn) ? 1 : 0;
      }
      *bestSplitToReturn = pack(numCategoriesForVar, icat);
    }
  } else *jstat = NODE_TERMINAL;

  R_Free(ncase);
  R_Free(varIndices);
  R_Free(v);
  R_Free(yl);
  R_Free(xt);
  R_Free(ut);
  R_Free(selected_groups);
  R_Free(group_vars);
  R_Free(temp_stratum);
}