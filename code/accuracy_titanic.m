## Copyright (C) 2017 MichaelEddy
## 
## Returns the accuracy given a threshold, y_hat and y. 

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} predict_titanic (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: MichaelEddy <MichaelEddy@LAPTOP-D5MV596A>
## Created: 2017-12-25

function [acc] = accuracy_titanic(t, y_hat, y)

  temp = y_hat>t;
  [m, n] = size(y);
  acc = sum(temp==y)/m;

endfunction
