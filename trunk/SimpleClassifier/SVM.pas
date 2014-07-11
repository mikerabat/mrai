// ###################################################################
// #### This file is part of the artificial intelligence project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2014, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################

unit SVM;

// ########################################################
// #### Simple Support Vector Machine class
// ########################################################

// following the example in the algorithm:
// "Pattern Classification" - Stork et al

// the Lagragian part is based on:
// Mangasarian and Musicant: "Lagragian Support Vector Machines"

// the classifier can also process simple Matlab generated classifier.

interface

uses SysUtils, Classes, Types, BaseClassifier, BaseMathPersistence, Matrix;

type
  TSVMLearnMethod = (lmLeastSquares, lmLagrangian);     
  TSVMKernel = (svmPoly, svmPolyInhomogen, svmGauss, svmSigmoid);
  TSVMProps = record
    learnMethod : TSVMLearnMethod;
    slack : double;     // see Lagragian support vector machines
    autoScale : boolean;
    case kernelType : TSVMKernel of
      svmPoly,
      svmPolyInhomogen: ( order : integer; );
      svmGauss : ( sigma : double; );

      svmSigmoid : ( offset : double;
                     scale : double; );
  end;

// ######################################################
// #### Classifies the example according
type
  TSVMClassifier = class(TCustomClassifier)
  private
    fW : IMatrix;        // classification plane (basically a weighting of the support vectors)
    fSV : IMatrix;       // support vectors
    fBias : IMatrix;
    fScaleMean : IMatrix;// mean and stdev scaling
    fScaleFact : IMatrix;
    fProps : TSVMProps;
    fClassVal1, fClassVal2 : integer;

    function AugmentedExample(Example : TCustomExample) : IMatrix;
    function PolyKernel(Example : TCustomExample) : IMatrix;
    function SigmoidKernel(Example : TCustomExample) : IMatrix;
    function GaussKernel(Example : TCustomExample) : IMatrix;
    procedure PolyFunc(var value: double);
    procedure PolyFuncInhom(var value: double);
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; overload; override;
    function Classify(Example : TCustomExample) : integer; overload; override;

    // loading and saving procedures
    function OnLoadObject(const Name : String; Obj : TBaseMathPersistence) : boolean; overload; override;
    procedure OnLoadIntProperty(const Name : String; Value : integer); override;
    procedure OnLoadDoubleProperty(const Name : String; const Value : double); override;

    procedure DefineProps; override;
    constructor Create(const props : TSVMProps; w, sv, bias : IMatrix; c1, c2 : integer; scaleMean, scaleFact : IMatrix);
  end;

// #####################################################
// #### Support vector machine learning class
type
  ESVMLearnerException = class(ECustomClassifierException);
  ESVMClassException = class(ESVMLearnerException);
  TSVMLearner = class(TCustomWeightedLearner)
  private
    fProps : TSVMProps;
    fZIdx : TIntegerDynArray;

    faStar : IMatrix;
    fBias : IMatrix;

    fVectIdx : TIntegerDynArray;

    function PolyKernelData(augTrainSet : IMatrix) : IMatrix;
    function GaussKernelData(augTrainSet : IMatrix) : IMatrix;
    function SigmoidKernelData(augTrainSet : IMatrix) : IMatrix;
    function TrainSet(doAugment : boolean) : IMatrix;

    procedure LearnLagrangian(y : IMatrix); // this algorithm is not very good... (numerically instable + very bad results)
    procedure LearnLeastSquares(y : IMatrix);
  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure AfterConstruction; override;

    procedure SetProps(const Props : TSVMProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses BaseMatrixExamples, MatrixConst, MathUtilFunc, math;

procedure TanhFunc(var value : double);
begin
     value := tanh(value);
end;

{ TSVMLearner }

procedure TSVMLearner.AfterConstruction;
begin
     inherited;

     fProps.learnMethod := lmLagrangian;
     fProps.kernelType := svmPoly;
     fProps.order := 2;
     fProps.slack := 1;
end;

function TSVMLearner.TrainSet(doAugment : boolean): IMatrix;
var x, y : integer;
    augsize : integer;
begin
     augsize := 0;
     if doAugment then
        augsize := 1;

     Result := TDoubleMatrix.Create(DataSet.Count, DataSet.Example[0].FeatureVec.FeatureVecLen + augsize, 1);

     if DataSet is TMatrixLearnerExampleList
     then
         Result.AssignSubMatrix(TMatrixLearnerExampleList(DataSet).Matrix, 0, 0)
     else
     begin
          // Create a matrix of the feature vectors -> this classifier only understands matrices
          Result := TDoubleMatrix.Create(DataSet.Count, DataSet[0].FeatureVec.FeatureVecLen);
          for y := 0 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
              for x := 0 to DataSet.Count - 1 do
                  Result[x, y] := DataSet[x].FeatureVec[y];
     end;
end;

class function TSVMLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TSVMClassifier;
end;

function TSVMLearner.DoLearn(const weights: array of double): TCustomClassifier;
var augExmpl : IMatrix;
    y : IMatrix;
    idx : TIntIntArray;
    numCl : integer;
    cl : TIntegerDynArray;
    sv : IMatrix;
    counter: Integer;
    scaleMean : IMatrix;
    scaleFact : IMatrix;
    ones : IMatrix;
    tmp : IMatrix;
begin
     numCl := IndexOfClasses(idx, cl);
     if numCl <> 2 then
        raise ESVMClassException.Create('Error only 2 class problems are supported');

     // store indices for faster diagonal multiplication
     if fProps.learnMethod = lmLeastSquares
     then
         fZIdx := idx[0]
     else
         fZIdx := idx[1];

     // ###########################################
     // #### augmented training set
     augExmpl := TrainSet(fProps.learnMethod = lmLagrangian);

     scaleMean := nil;
     scaleFact := nil;
     
     // ###########################################
     // #### Data scaling
     if fProps.autoScale then
     begin
          if fProps.learnMethod = lmLagrangian then
             augExmpl.SetSubMatrix(0, 0, augExmpl.Width - 1, augExmpl.Height);
          // calculate autoscale parameters:
          // use standard deviation and mean
          scaleMean := augExmpl.Mean(True);
          scaleFact := augExmpl.Std(True);

          // scaling of "infinity" or zero shall be 1
          for counter := 0 to scaleFact.Height - 1 do
          begin
               if (scaleFact[0, counter] = Infinity) or (scaleFact[0, counter] = 0) then
                  scaleFact[0, counter] := 1;
          end;

          ones := TDoubleMatrix.Create(scaleFact.Width, scaleFact.Height, 1);
          scaleFact := ones.ElementWiseDiv(scaleFact);

          // now scale the examples!
          for counter := 0 to scaleMean.Height - 1 do
          begin
               augExmpl.SetSubMatrix(0, counter, augExmpl.Width, 1);
               augExmpl.AddAndScaleInPlace(-scaleMean[0, counter], scaleFact[0, counter]);
          end;

          augExmpl.UseFullMatrix;
     end;
     
     // ###########################################
     // #### Applying the kernel
     case fProps.kernelType of
       svmPolyInhomogen,
       svmPoly: y := PolyKernelData(augExmpl);
       svmGauss: y := GaussKernelData(augExmpl);
       svmSigmoid: y := SigmoidKernelData(augExmpl);
     end;

     // ###########################################
     // #### Learn classifier
     case fProps.learnMethod of
       lmLeastSquares: LearnLeastSquares(y);
       lmLagrangian: LearnLagrangian(y);
     end;

     //  now build the support vector matrix and reduce the weighting matrix
     sv := TDoubleMatrix.Create(Length(fVectIdx), augExmpl.Height);

     for counter := 0 to Length(fVectIdx) - 1 do
     begin
          faStar[0, counter] := faStar[0, fVectIdx[counter]];
          sv.SetColumn(counter, augExmpl, fVectIdx[counter]);
     end;

     faStar.UseFullMatrix;
     if Length(fVectIdx) < faStar.Height then
     begin
          faStar.SetSubMatrix(0, 0, 1, Length(fVectIdx));
          tmp := TDoubleMatrix.Create;
          tmp.Assign(faStar, True);
          faStar := tmp;
     end;

     // ##############################################
     // #### Construct classifier object
     Result := TSVMClassifier.Create(fProps, faStar, sv, fBias, cl[0], cl[1], scaleMean, scaleFact);
end;

procedure TSVMLearner.LearnLagrangian(y : IMatrix);
var tol : double;
    maxIter : integer;
    nu  : double;
    iter : integer;
    alpha : double;
    e : IMatrix;
    Q : IMatrix;
    P : IMatrix;
    u : IMatrix;
    oldu : IMatrix;
    f : IMatrix;
    cnt : integer;
    nuInv : double;
    utmp : IMatrix;
    numVec : integer;

function uDiff : double;
var u1 : IMatrix;
begin
     u1 := u.Sub(oldu);
     Result := u1.ElementwiseNorm2;
end;
begin
     tol := 1e-5;
     maxIter := 100000;
     nu := 1/DataSet.Count;
     nuInv := DataSet.Count;

     alpha := 1.9/nu;
     e := TDoubleMatrix.Create(1, DataSet.Count, 1);

     // calculate Q = speye(Nf)/nu + D*y'*D
     // D is a diagonal matrix having -1, 1 values according to the class labels
     Q := y.Transpose;
     for cnt := 0 to Length(fZIdx) - 1 do
     begin
          Q.SetSubMatrix(0, fZIdx[cnt], Q.Width, 1);
          Q.ScaleInPlace(-1);
     end;
     Q.UseFullMatrix;
     for cnt := 0 to Length(fZIdx) - 1 do
     begin
          Q.SetSubMatrix(fZIdx[cnt], 0, 1, Q.Height);
          Q.ScaleInPlace(-1);
     end;
     Q.UseFullMatrix;
     for cnt := 0 to y.Height - 1 do
         Q[cnt, cnt] := Q[cnt, cnt] + nuInv;

     // u=P*e;
     // oldu=u+1
     P := Q.Invert;
     u := P.Mult(e);
     oldu := u.Add(1);

     // main loop
     iter := 0;

     while (iter < maxIter) and (uDiff > tol) do
     begin
          // matlab code:
          // oldu    = u;
          // f       = Q*u-1-alpha*u;
          // u       = P*(1+(abs(f)+f)/2);
          oldu := u;

          utmp := u.Scale(alpha);

          f := Q.Mult(u);
          f.AddInPlace(-1);
          f.SubInPlace(utmp);

          utmp := f.Abs;
          f.AddInplace(utmp);
          f.ScaleAndAddInPlace(1, 0.5);
          u := P.Mult(f);

          inc(iter);
     end;

     // Matlab code after loop:
     // getting support vectors
     // a_star    = y*D*u(1:Nf);
     // bias      = -e'*D*u;
     // sv		  = find(abs(a_star) > slack*1e-3);
     for cnt := 0 to Length(fZIdx) - 1 do
     begin
          y.SetSubMatrix(fZIdx[cnt], 0, 1, y.Height);
          y.ScaleInPlace(-1);
     end;

     y.UseFullMatrix;
     y.MultInPlace(u);
     faStar := y;

     for cnt := 0 to Length(fZIdx) - 1 do
     begin
          u.SetSubMatrix(0, fZIdx[cnt], u.Width, 1);
          u.ScaleInPlace(-1);
     end;
     u.UseFullMatrix;
     fBias := e.Transpose;
     fBias.ScaleInPlace(-1);
     fBias.MultInPlace(u);

     // find support vectors
     SetLength(fVectIdx, faStar.Height);
     numVec := 0;
     for cnt := 0 to faStar.Height - 1 do
     begin
          if Abs(faStar[0, cnt]) >= fProps.slack*0.001  then
          begin
               fVectIdx[NumVec] := cnt;
               inc(NumVec);
          end
          else
              faStar[0, cnt] := 0;   // we do not want any component from a very small indices
     end;

     SetLength(fVectIdx, NumVec);
end;

procedure TSVMLearner.LearnLeastSquares(y: IMatrix);
var kx : IMatrix;
    grIndex : TDoubleDynArray;
    counter: Integer;
    A : IMatrix;
    b : IMatrix;
    numElem : integer;
begin
     // ensure function is symmetric
     //kx = (kx+kx')/2 + diag(1./boxconstraint);
     kx := y.Transpose;
     kx.AddInplace(y);
     kx.ScaleInPlace(1/2);

     for counter := 0 to kx.Width - 1 do
         kx[counter, counter] := kx[counter, counter] + 1;

     // create hessian
     numElem := DataSet.Count - Length(fZIdx);
     kx.SetSubMatrix(0, kx.Height - numElem, numElem, kx.Height - numElem);
     kx.ScaleInPlace(-1);
     kx.UseFullMatrix;
     kx.SetSubMatrix(numElem, 0, kx.Width - numElem, numElem);
     kx.ScaleInPlace(-1);
     kx.UseFullMatrix;

     // create augmented matrix A
     A := TDoubleMatrix.Create(DataSet.Count + 1, DataSet.Count + 1);
     SetLength(grIndex, DataSet.Count + 1);
     for counter := 0 to Length(fZIdx) - 1 do
         grIndex[fZIdx[counter] + 1] := 1;

     for counter := 1 to Length(grIndex) - 1 do
         if grIndex[counter] <> 1 then
            grIndex[counter] := -1;

     A.SetRow(0, grIndex);
     A.SetColumn(0, grIndex);
     A.SetSubMatrix(1, 1, kx.Width, kx.Height);
     A.AddInplace(kx);
     kx := nil;
     A.UseFullMatrix;

     b := TDoubleMatrix.Create(1, A.Height, 1);
     b[0, 0] := 0;

     try
        A.SolveLinEQInPlace(b);
     except
           // standard LU decomposition failed -> use pseudoinverse
           if A.PseudoInversionInPlace = srNoConvergence then
              raise ESVMLearnerException.Create('Error inversion failed');

           A.MultInPlace(b);
     end;

     fBias := TDoubleMatrix.Create(1, 1, A[0,0]);

     SetLength(fVectIdx, DataSet.Count);
     for counter := 0 to Length(fVectIdx) - 1 do
         fVectIdx[counter] := counter;

     faStar := TDoubleMatrix.Create(1, A.Height - 1);
     
     for counter := 0 to faStar.Height - 1 do
         faStar[0, counter] := A[0, counter + 1]*grIndex[counter + 1];
end;

function TSVMLearner.PolyKernelData(augTrainSet : IMatrix): IMatrix;
var x : Integer;
    tTrainSet : IMatrix;
    dotproduct : IMatrix;
begin
     tTrainSet := augTrainSet.Transpose;

     dotproduct := tTrainSet.Mult(augTrainSet);
     Result := TDoubleMatrix.Create;
     Result.Assign(dotproduct);

     if fProps.kernelType = svmPolyInhomogen then
        dotproduct.AddInplace(1);

     for x := 1 to fProps.order - 1 do
         Result.ElementWiseMultInPlace(dotproduct);         
end;

function TSVMLearner.SigmoidKernelData(augTrainSet: IMatrix): IMatrix;
var x : Integer;
    tTrainSet : IMatrix;
    col : IMatrix;
begin
     Result := TDoubleMatrix.Create(augTrainSet.Width, augTrainSet.Width);
     tTrainSet := augTrainSet.Transpose;

     for x := 0 to Result.Height - 1 do
     begin
          augTrainSet.SetSubMatrix(x, 0, 1, augTrainSet.Height);
          col := tTrainSet.Mult(augTrainSet);

          col.ScaleAndAddInPlace(fProps.scale, fProps.offset);
          col.ElementwiseFuncInPlace(TanhFunc);

          Result.SetColumn(x, col);
     end;

     augTrainSet.UseFullMatrix;
end;

function TSVMLearner.GaussKernelData(augTrainSet: IMatrix): IMatrix;
var gamma : double;
    i, j : integer;
    xi, sum : IMatrix;
    val : double;
begin
     Result := TDoubleMatrix.Create(augTrainSet.Width, augTrainSet.Width);

     // y(:,i)    = exp(-sum((train_patterns-train_patterns(:,i)*ones(1,Nf)).^2)'/(2*ker_param^2));
     // according to wikipedia k(xi, xj) = exp(-s * ||xi - xj]]^2)   s = 1/(2*sigma^2);
     gamma := -1/(2*sqr(fProps.sigma));
     xi := TDoubleMatrix.Create(1, augTrainset.Height);
     for i := 0 to augTrainSet.Width - 1 do
     begin
          xi.SetColumn(0, augTrainSet, i);
          Result[i, i] := 1;
          for j := i + 1 to augTrainSet.Width - 1 do
          begin
               augTrainSet.SetSubMatrix(j, 0, 1, augTrainSet.Height);
               sum := augTrainSet.Sub(xi);

               val := exp(gamma*sqr(sum.ElementwiseNorm2));

               Result[i, j] := val;
               Result[j, i] := val;
          end;

          augTrainSet.UseFullMatrix;
     end;
end;


procedure TSVMLearner.SetProps(const Props: TSVMProps);
begin
     fProps := Props;
end;

{ TSVMClassifer }

function TSVMClassifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var x : IMatrix;
begin
     // kernel evaluation
     case fProps.kernelType of
       svmPoly,
       svmPolyInhomogen : x := PolyKernel(Example);
       svmGauss: x := GaussKernel(Example);
       svmSigmoid: x := SigmoidKernel(Example);
     end;

     // weighting function (the classification plane)
     x.TransposeInPlace;
     x.MultInPlace(fW);
     x.AddInplace(fBias);

     // classify
     confidence := Abs(x[0, 0]);
     if x[0, 0] >= 0
     then
         Result := fClassVal1
     else
         Result := fClassVal2;
end;

function TSVMClassifier.AugmentedExample(Example: TCustomExample): IMatrix;
var y : integer;
begin
     Result := TDoubleMatrix.Create(1, fSV.Width, 1);

     // Create a matrix of the feature vectors -> this classifier only understands matrices
     for y := 0 to Example.FeatureVec.FeatureVecLen - 1 do
         Result[0, y] := Example.FeatureVec[y];

     // ###########################################
     // #### Autoscaling according to the train set
     if fProps.autoScale and Assigned(fScaleMean) and Assigned(fScaleFact) then
     begin
          Result.SetSubMatrix(0, 0, 1, fScaleMean.Height);
          Result.SubInPlace(fScaleMean);
          Result.ElementWiseMultInPlace(fScaleFact);
          Result.UseFullMatrix;
     end;
end;

function TSVMClassifier.Classify(Example: TCustomExample): integer;
var conf : double;
begin
     Result := Classify(Example, conf);
end;

constructor TSVMClassifier.Create(const props: TSVMProps; w, sv, bias: IMatrix; c1, c2 : integer; scaleMean, scaleFact : IMatrix);
begin
     inherited Create;

     fScaleMean := scaleMean;
     fScaleFact := scaleFact;
     fProps := props;
     fW := w;
     fBias := bias;
     fSV := sv.Transpose;
     fClassVal1 := c1;
     fClassVal2 := c2;
end;

procedure TSVMClassifier.PolyFunc(var value: double);
var cnt : integer;
    mul : double;
begin
     mul := value;
     for cnt := 0 to fProps.order - 2 do
         value := value*mul;
end;


procedure TSVMClassifier.PolyFuncInhom(var value: double);
var cnt : integer;
    mul : double;
begin
     mul := value + 1;
     for cnt := 0 to fProps.order - 2 do
         value := value*mul;
end;

function TSVMClassifier.PolyKernel(Example: TCustomExample): IMatrix;
var k : IMatrix;
begin
     k := AugmentedExample(Example);

     Result := fSV.Mult(k);

     if fProps.order > 1 then
     begin
          if fProps.kernelType = svmPolyInhomogen
          then
              Result.ElementwiseFuncInPlace(PolyFuncInhom)
          else
              Result.ElementwiseFuncInPlace(PolyFunc);
     end;
end;

function TSVMClassifier.SigmoidKernel(Example: TCustomExample): IMatrix;
var k : IMatrix;
begin
     k := AugmentedExample(Example);

     Result := fSV.Mult(k);

     Result.AddAndScaleInPlace(fProps.scale, fProps.offset);
     Result.ElementwiseFuncInPlace(TanhFunc);
end;

function TSVMClassifier.GaussKernel(Example: TCustomExample): IMatrix;
var k : IMatrix;
    gamma : double;
    i : integer;
    sum : IMatrix;
begin
     Result := TDoubleMatrix.Create(1, fSV.Height);
     k := AugmentedExample(Example);
     // note we need to transpose the example since the support vectors are transposed as well
     k.TransposeInPlace;

     gamma := -1/(2*sqr(fProps.sigma));
     for i := 0 to fSV.Height - 1 do
     begin
          fSV.SetSubMatrix(0, i, fSV.Width, 1);

          sum := fSV.Sub(k);
          Result[0, i] := exp(gamma*sqr(sum.ElementwiseNorm2));
          fSV.UseFullMatrix;
     end;
end;

// ###################################################
// #### Load/store
// ###################################################

procedure TSVMClassifier.DefineProps;
begin
     inherited;

     AddObject('SVMW', fW.GetObjRef);
     AddObject('SVMSV', fSV.GetObjRef);
     AddObject('SVMBIAS', fBias.GetObjRef);

     AddIntProperty('SVMClass1', fClassVal1);
     AddIntProperty('SVMClass2', fClassVal2);

     AddIntProperty('SVMAutoScale', Integer(fProps.autoScale));
     AddDoubleProperty('SVMSlack', fProps.slack);

     if Assigned(fScaleMean) then
        AddObject('SVMScaleMean', fScaleMean.GetObjRef);
     if Assigned(fScaleFact) then
        AddObject('SVMScaleFact', fScaleFact.GetObjRef);
     
     AddIntProperty('SVMLearnMethod', Integer(fProps.learnMethod));
     AddIntProperty('SVMKernelType', Integer(fProps.kernelType));

     case fProps.kernelType of
       svmPoly,
       svmPolyInhomogen: AddIntProperty('SVMPolyOrder', fProps.order);
       svmGauss : AddDoubleProperty('SVMGaussSigma', fProps.sigma);
       svmSigmoid : begin
                         AddDoubleProperty('SVMSigmoidOffset', fProps.offset);
                         AddDoubleProperty('SVMSigmoidScale', fProps.scale);
                    end;
     end;
end;

procedure TSVMClassifier.OnLoadDoubleProperty(const Name: String;
  const Value: double);
begin
     if SameText(Name, 'SVMSlack')
     then
         fProps.slack := Value
     else if SameText(Name, 'SVMGaussSigma')
     then
         fProps.sigma := Value
     else if SameText(Name, 'SVMSigmoidOffset')
     then
         fProps.offset := Value
     else if SameText(Name, 'SVMSigmoidScale')
     then
         fProps.scale := Value
     else
         inherited;
end;

procedure TSVMClassifier.OnLoadIntProperty(const Name: String; Value: integer);
begin
     if SameText(Name, 'SVMClass1')
     then
         fClassVal1 := Value
     else if SameText(Name, 'SVMClass2')
     then
         fClassVal2 := Value
     else if SameText(Name, 'SVMAutoScale')
     then
         fProps.autoScale := Value <> 0
     else if SameText(Name, 'SVMLearnMethod')
     then
         fProps.learnMethod := TSVMLearnMethod(Value)
     else if SameText(Name, 'SVMKernelType')
     then
         fProps.kernelType := TSVMKernel(Value)
     else
         inherited;
end;

function TSVMClassifier.OnLoadObject(const Name: String;
  Obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if SameText(Name, 'SVMW')
     then
         fW := obj as TDoubleMatrix
     else if SameText(Name, 'SVMSV')
     then
         fSV := obj as TDoubleMatrix
     else if SameText(Name, 'SVMBIAS')
     then
         fBias := obj as TDoubleMatrix
     else if SameText(Name, 'SVMScaleMean' )
     then 
          fScaleMean := obj as TDoubleMatrix
     else if SameText(Name, 'SVMScaleFact')
     then
         fScaleFact := obj as TDoubleMatrix
     else
         Result := inherited OnLoadObject(Name, Obj);
end;

end.
