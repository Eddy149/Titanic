## Copyright (C) 2017 MichaelEddy
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} predict_titanicNN (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: MichaelEddy <MichaelEddy@LAPTOP-D5MV596A>
## Created: 2017-12-25

function [y] = predict_titanicNN (X, Theta1, Theta2, Theta3)
  [m n] = size(X);
  a1 = [ones(m,1), X];
  z2 = a1*Theta1';
  a2 = sigmoid(z2);
  a2 = [ones(size(a2,1), 1), a2]; 

  z3 = a2 * Theta2';
  a3 = sigmoid(z3);
  a3 = [ones(size(a2,1), 1), a3]; 

  z4 = a3 * Theta3';
  y = sigmoid(z4);  
endfunction
