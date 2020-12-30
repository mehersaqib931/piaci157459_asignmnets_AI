{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NgUeapcvmIRh"
   },
   "source": [
    "# **Assignment For Numpy**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yhr3JicwnA-4"
   },
   "source": [
    "Difficulty Level **Beginner**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hVPqDJ1SnKSh"
   },
   "source": [
    "1. Import the numpy package under the name np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "SePu31zKmgS-"
   },
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d0gO81krnL7g"
   },
   "source": [
    "2. Create a null vector of size 10 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "2s6o2JPgnSBr"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.zeros(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BJTMK03fnS1w"
   },
   "source": [
    "3. Create a vector with values ranging from 10 to 49"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "qRXxXuWdnj1M"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33\n",
      " 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49]\n"
     ]
    }
   ],
   "source": [
    "a=np.arange(10,50)\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "n63lzg5Onkcn"
   },
   "source": [
    "4. Find the shape of previous array in question 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "1FueBdqPn_q7"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40,)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.shape(a)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "AzMlQj1MoAVe"
   },
   "source": [
    "5. Print the type of the previous array in question 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "AO9PYmdWoE1L"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'numpy.ndarray'>\n"
     ]
    }
   ],
   "source": [
    "print(type(a))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0qaKS13Yon-U"
   },
   "source": [
    "6. Print the numpy version and the configuration\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "T3wY24e1oorh"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.18.1\n",
      "blas_mkl_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/saqib/anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/saqib/anaconda3\\\\Library\\\\include']\n",
      "blas_opt_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/saqib/anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/saqib/anaconda3\\\\Library\\\\include']\n",
      "lapack_mkl_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/saqib/anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/saqib/anaconda3\\\\Library\\\\include']\n",
      "lapack_opt_info:\n",
      "    libraries = ['mkl_rt']\n",
      "    library_dirs = ['C:/Users/saqib/anaconda3\\\\Library\\\\lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\include', 'C:\\\\Program Files (x86)\\\\IntelSWTools\\\\compilers_and_libraries_2019.0.117\\\\windows\\\\mkl\\\\lib', 'C:/Users/saqib/anaconda3\\\\Library\\\\include']\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(np.__version__)\n",
    "print(np.show_config())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZyUUiQf5oyvE"
   },
   "source": [
    "7. Print the dimension of the array in question 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "pLXfuIqIo0vq"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1\n"
     ]
    }
   ],
   "source": [
    "print(np.ndim(a))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JYVMuFrqpBdV"
   },
   "source": [
    "8. Create a boolean array with all the True values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "3apZsISzpFKR"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ True  True  True  True  True]\n"
     ]
    }
   ],
   "source": [
    "arr=[2,3,4,5,6]\n",
    "arr1=np.array(arr)\n",
    "bool_arr=arr1>1\n",
    "print(bool_arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4zbBooWZpPBU"
   },
   "source": [
    "9. Create a two dimensional array\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "KfPQEiVIpdTo"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 2, 3],\n",
       "       [4, 5, 6]])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.array([[1,2,3],[4,5,6]])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "do9wPAFbpqC7"
   },
   "source": [
    "10. Create a three dimensional array\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "ZCO_q-KZqAeW"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 0,  1,  2],\n",
       "        [ 3,  4,  5],\n",
       "        [ 6,  7,  8]],\n",
       "\n",
       "       [[ 9, 10, 11],\n",
       "        [12, 13, 14],\n",
       "        [15, 16, 17]],\n",
       "\n",
       "       [[18, 19, 20],\n",
       "        [21, 22, 23],\n",
       "        [24, 25, 26]]])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.arange(0,27).reshape(3,3,3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "a4iysJ87qEeb"
   },
   "source": [
    "Difficulty Level **Easy**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DehgqszSqjY6"
   },
   "source": [
    "11. Reverse a vector (first element becomes last)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "tfevmc4VqkWu"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr=np.arange(10)\n",
    "arr[::-1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yoATXS-qqk_q"
   },
   "source": [
    "12. Create a null vector of size 10 but the fifth value which is 1 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "AheZ32Owqpte"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "arr=np.zeros(10)\n",
    "arr[4]=1\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RZRzOBbFsY0w"
   },
   "source": [
    "13. Create a 3x3 identity matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "id": "va2ou0BHsvEj"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 1 1]\n",
      " [1 1 1]\n",
      " [1 1 1]]\n"
     ]
    }
   ],
   "source": [
    "arr=np.ones(9,dtype=int).reshape(3,3)\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lnN5drkUs6o2"
   },
   "source": [
    "14. arr = np.array([1, 2, 3, 4, 5]) \n",
    "\n",
    "---\n",
    "\n",
    " Convert the data type of the given array from int to float "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "id": "59EIQt6otKUB"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1. 2. 3. 4. 5.]\n"
     ]
    }
   ],
   "source": [
    "arr = np.array([1, 2, 3, 4, 5],dtype=float)\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fSL2AKJetTes"
   },
   "source": [
    "15. arr1 =          np.array([[1., 2., 3.],\n",
    "\n",
    "                    [4., 5., 6.]])  \n",
    "                      \n",
    "    arr2 = np.array([[0., 4., 1.],\n",
    "     \n",
    "                   [7., 2., 12.]])\n",
    "\n",
    "---\n",
    "\n",
    "\n",
    "Multiply arr1 with arr2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "7VDXgRQDuJNV"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.,  8.,  3.],\n",
       "       [28., 10., 72.]])"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1 = np.array([[1., 2., 3.],\n",
    "\n",
    "            [4., 5., 6.]])  \n",
    "arr2 = np.array([[0., 4., 1.],\n",
    "\n",
    "           [7., 2., 12.]])\n",
    "arr1 * arr2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "a8yJ438-uKni"
   },
   "source": [
    "16. arr1 = np.array([[1., 2., 3.],\n",
    "                    [4., 5., 6.]]) \n",
    "                    \n",
    "    arr2 = np.array([[0., 4., 1.], \n",
    "                    [7., 2., 12.]])\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "Make an array by comparing both the arrays provided above"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "3MPDaAlYueF3"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[False False False]\n",
      " [False False False]]\n"
     ]
    }
   ],
   "source": [
    "rr1 = np.array([[1., 2., 3.],\n",
    "\n",
    "            [4., 5., 6.]]) \n",
    "arr2 = np.array([[0., 4., 1.],\n",
    "\n",
    "            [7., 2., 12.]])\n",
    "comp_arr=arr1 ==arr2\n",
    "print(comp_arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3xiD0eJluewk"
   },
   "source": [
    "17. Extract all odd numbers from arr with values(0-9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "J_qbt_EuvM7l"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 3, 5, 7, 9])"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr=np.arange(10)\n",
    "arr[arr%2!=0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zpg5cyLsvPoZ"
   },
   "source": [
    "18. Replace all odd numbers to -1 from previous array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "rXUV1-ULvQdd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0 -1  2 -1  4 -1  6 -1  8 -1]\n"
     ]
    }
   ],
   "source": [
    "arr=np.arange(10)\n",
    "arr[arr%2!=0]=-1\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FWpHTWTDvjvi"
   },
   "source": [
    "19. arr = np.arange(10)\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "Replace the values of indexes 5,6,7 and 8 to **12**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "id": "ensZ3lqwvlSb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0  1  2  3  4 12 12 12  8  9]\n"
     ]
    }
   ],
   "source": [
    "arr=np.arange(10)\n",
    "arr[[5,6,7]]=12\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ib-vBffAv0zy"
   },
   "source": [
    "20. Create a 2d array with 1 on the border and 0 inside"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "id": "RmzcjNYrwDrM"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 1. 1. 1. 1. 1.]\n",
      " [1. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 1.]\n",
      " [1. 0. 0. 0. 0. 1.]\n",
      " [1. 1. 1. 1. 1. 1.]]\n"
     ]
    }
   ],
   "source": [
    "arr=np.ones((6,6))\n",
    "arr[1:-1, 1:-1]=0\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "E1diIFSEwEmh"
   },
   "source": [
    "Difficulty Level **Medium**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "eGLZmNb_wKb4"
   },
   "source": [
    "21. arr2d = np.array([[1, 2, 3],\n",
    "\n",
    "                    [4, 5, 6], \n",
    "\n",
    "                    [7, 8, 9]])\n",
    "\n",
    "---\n",
    "\n",
    "Replace the value 5 to 12"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "VdUnsUOZwJFz"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  2  3]\n",
      " [ 4 12  6]\n",
      " [ 7  8  9]]\n"
     ]
    }
   ],
   "source": [
    "arr2d = np.array([[1, 2, 3],\n",
    "\n",
    "            [4, 5, 6], \n",
    "\n",
    "            [7, 8, 9]])\n",
    "arr2d[1][1]=12\n",
    "print(arr2d)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TEbFhlPZxaKO"
   },
   "source": [
    "22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])\n",
    "\n",
    "---\n",
    "Convert all the values of 1st array to 64\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "id": "eYzephU8xmsg"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[64 64 64]\n",
      "  [64 64 64]]\n",
      "\n",
      " [[ 7  8  9]\n",
      "  [10 11 12]]]\n"
     ]
    }
   ],
   "source": [
    "arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])\n",
    "arr3d[0]=64\n",
    "print(arr3d)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vZj0Ndvnx6f0"
   },
   "source": [
    "23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "id": "pM4e9YfdyHSv"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr=np.arange(0,9).reshape(3,3)\n",
    "print(arr)\n",
    "arr[0,:]\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "C3dkW0lZyVps"
   },
   "source": [
    "24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "id": "IrrzK2xgygqV"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "4"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr=np.arange(0,9).reshape(3,3)\n",
    "print(arr)\n",
    "arr[1, 1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "G3dLwOO9yyrE"
   },
   "source": [
    "25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "id": "5Bx4fqsLyqGD"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([2, 5])"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr=np.arange(0,9).reshape(3,3)\n",
    "print(arr)\n",
    "arr[0:2, 2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CQ9YST5jy0IL"
   },
   "source": [
    "26. Create a 10x10 array with random values and find the minimum and maximum values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "id": "lhJPugjQzG6W"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.13451977 0.21334358 0.06520006 0.14172089 0.02119755 0.80066591\n",
      "  0.82700932 0.47277181 0.73621999 0.64215107]\n",
      " [0.28731077 0.69972866 0.19593772 0.33136672 0.70412206 0.574401\n",
      "  0.18434171 0.94873649 0.92387253 0.8562905 ]\n",
      " [0.26262175 0.97769149 0.84907983 0.03069648 0.31120481 0.94348681\n",
      "  0.57084066 0.84182217 0.31107096 0.8763137 ]\n",
      " [0.49101857 0.6677697  0.61373385 0.91127178 0.00406373 0.36354247\n",
      "  0.60398269 0.76382266 0.35082463 0.64786507]\n",
      " [0.40849093 0.33840678 0.73632703 0.25966074 0.30766667 0.96391495\n",
      "  0.8681979  0.33092426 0.11417093 0.67750503]\n",
      " [0.18304256 0.96674964 0.28878962 0.44215475 0.67017245 0.34595971\n",
      "  0.39694638 0.09815301 0.77057571 0.0208735 ]\n",
      " [0.78653951 0.18971182 0.36172728 0.28069576 0.76725378 0.98002164\n",
      "  0.56869743 0.13073595 0.78022649 0.08200785]\n",
      " [0.36130549 0.4186626  0.07989799 0.33513755 0.050574   0.27337307\n",
      "  0.93725877 0.02673529 0.55380981 0.20719058]\n",
      " [0.36714186 0.15462129 0.01975959 0.74206263 0.34255404 0.46926802\n",
      "  0.33575048 0.88381999 0.74073771 0.6253326 ]\n",
      " [0.84491882 0.70825463 0.01024219 0.19559703 0.30255464 0.14089934\n",
      "  0.59181483 0.6032335  0.11843848 0.29698708]]\n",
      "the miminim and maximumum values are\n",
      "0.9800216405708129 0.004063725156789211\n"
     ]
    }
   ],
   "source": [
    "arr=np.random.random((10,10))\n",
    "print(arr)\n",
    "minimum=arr.min()\n",
    "maximum=arr.max()\n",
    "print(\"the miminim and maximumum values are\")\n",
    "print(maximum, minimum)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cAgDt6KazHk6"
   },
   "source": [
    "27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "---\n",
    "Find the common items between a and b\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "id": "kBXZanxIzs_F"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[2 4]\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1,2,3,2,3,4,3,4,5,6]) \n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "com_arr=np.intersect1d(a,b)\n",
    "print(com_arr)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lF9IzWOAztpD"
   },
   "source": [
    "28. a = np.array([1,2,3,2,3,4,3,4,5,6])\n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "\n",
    "---\n",
    "Find the positions where elements of a and b match\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "id": "JR-mHQLH0HY_"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([1, 3, 5, 7], dtype=int64),)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,2,3,2,3,4,3,4,5,6]) \n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "np.where(a==b)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mMDiefpH0H1x"
   },
   "source": [
    "29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)\n",
    "\n",
    "---\n",
    "Find all the values from array **data** where the values from array **names** are not equal to **Will**\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "id": "gfc1Lpf81ZSR"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.02354941, -1.30968855, -0.23554492, -0.04160881],\n",
       "       [ 1.20971549, -0.24530011,  0.18503456, -2.27650558],\n",
       "       [-2.75848468, -0.82453854, -0.98036607, -0.09961373],\n",
       "       [ 0.76547391, -0.13221062, -1.00120199, -0.60586602],\n",
       "       [-1.10614933,  1.26285393,  0.71289732,  0.34705237]])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) \n",
    "data = np.random.randn(7, 4)\n",
    "data[~(names==\"Will\")]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "52zEjDMf1aXS"
   },
   "source": [
    "30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)\n",
    "\n",
    "---\n",
    "Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "id": "8RYG0kLO1mHw"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.09670563,  1.27636968, -0.85226994, -0.01830451],\n",
       "       [ 0.15999202,  0.99673525,  0.377154  , -1.52916732]])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) \n",
    "data = np.random.randn(7, 4)\n",
    "data[~(names==\"Will\") & ~(names==\"Joe\")]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Q7hjf2bd2dCY"
   },
   "source": [
    "Difficulty Level **Hard**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b48g7aRA2jVM"
   },
   "source": [
    "31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "id": "EbwfSCrW2f9_"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1.  2.  3.]\n",
      " [ 4.  5.  6.]\n",
      " [ 7.  8.  9.]\n",
      " [10. 11. 12.]\n",
      " [13. 14. 15.]]\n"
     ]
    }
   ],
   "source": [
    "arr_flo=np.arange(1,16,dtype=float).reshape(5,3)\n",
    "print(arr_flo)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Ei_nDHtO3qBk"
   },
   "source": [
    "32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "id": "phOxVpBM4M6D"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1.,  2.,  3.,  4.],\n",
       "        [ 5.,  6.,  7.,  8.]],\n",
       "\n",
       "       [[ 9., 10., 11., 12.],\n",
       "        [13., 14., 15., 16.]]])"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_flo=np.arange(1,17,dtype=float).reshape(2,2,4)\n",
    "arr_flo"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uyiQaMjA4d_x"
   },
   "source": [
    "33. Swap axes of the array you created in Question 32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "id": "w2m9p8064VtR"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1.,  2.,  3.,  4.],\n",
       "        [ 9., 10., 11., 12.]],\n",
       "\n",
       "       [[ 5.,  6.,  7.,  8.],\n",
       "        [13., 14., 15., 16.]]])"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.swapaxes(arr_flo, 0,1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2dlU_yUR4mVZ"
   },
   "source": [
    "34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "id": "tmBabEJ-5JgB"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.         1.         1.41421356 1.73205081 2.         2.23606798\n",
      " 2.44948974 2.64575131 2.82842712 3.         3.16227766]\n"
     ]
    }
   ],
   "source": [
    "arr=np.arange(11)\n",
    "arr_sq=np.sqrt(arr)\n",
    "arr_sq[arr_sq<0.5]=0\n",
    "print(arr_sq)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SNAM4dpu5RKA"
   },
   "source": [
    "35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "id": "3pxebU2b5q9Z"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "9\n",
      "<class 'numpy.int32'>\n"
     ]
    }
   ],
   "source": [
    "arr1=np.random.randint(12)\n",
    "arr2=np.random.randint(12)\n",
    "arr=np.maximum(arr1,arr2)\n",
    "print(arr)\n",
    "print(type(arr))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fz_bpqm28qmD"
   },
   "source": [
    "36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "\n",
    "---\n",
    "Find the unique names and sort them out!\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "id": "hiSNMgcY87Ey"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Bob' 'Joe' 'Will']\n",
      "sorting\n",
      "['Bob' 'Joe' 'Will']\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "names_unique=np.unique(names)\n",
    "print(names_unique)\n",
    "print(\"sorting\")\n",
    "names_unique.sort()\n",
    "print(names_unique)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dguMNzob870H"
   },
   "source": [
    "37. a = np.array([1,2,3,4,5])\n",
    "b = np.array([5,6,7,8,9])\n",
    "\n",
    "---\n",
    "From array a remove all items present in array b\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "id": "eCNJXPvK9IAr"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4]\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1,2,3,4,5]) \n",
    "b = np.array([5,6,7,8,9])\n",
    "arr=np.setdiff1d(a,b)\n",
    "print(arr)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gz4Dvd4b_Akh"
   },
   "source": [
    "38.  Following is the input NumPy array delete column two and insert following new column in its place.\n",
    "\n",
    "---\n",
    "sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) \n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "newColumn = numpy.array([[10,10,10]])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "id": "lLWAJWJJ_I3d"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[34 43 73]\n",
      " [10 10 10]\n",
      " [53 94 66]]\n"
     ]
    }
   ],
   "source": [
    "import numpy\n",
    "sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])\n",
    "\n",
    "newColumn = numpy.array([[10,10,10]])\n",
    "sampleArray[1]=newColumn\n",
    "print(sampleArray)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1lIfmbQt_J_W"
   },
   "source": [
    "39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])\n",
    "\n",
    "\n",
    "---\n",
    "Find the dot product of the above two matrix\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "id": "fV2-vXcR_vmu"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 28.,  64.],\n",
       "       [ 67., 181.]])"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([[1., 2., 3.], [4., 5., 6.]]) \n",
    "y = np.array([[6., 23.], [-1, 7], [8, 9]])\n",
    "x.dot(y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5U-4odUw_wP0"
   },
   "source": [
    "40. Generate a matrix of 20 random values and find its cumulative sum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "id": "_oOHdiYEAF8N"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0.45393991, -0.53148773, -0.29605821, -0.2308922 , -0.88705676,\n",
       "        0.32218877, -0.80323807, -1.94117335, -1.21156426,  0.13401687,\n",
       "       -0.27690215, -1.57188932, -3.32741666, -2.89172609, -1.22921858,\n",
       "       -1.57449648, -1.76500373, -2.54785154, -2.0585759 , -2.44022279])"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr=np.random.randn(20)\n",
    "np.cumsum(arr, axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "name": "Numpy-Assignment-PIAIC-Batch35-Q2.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
