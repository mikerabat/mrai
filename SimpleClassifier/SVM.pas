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
// code is based on the lagragian implementation in the Pattern Classification book (svm.m)

// Least squares example is inspired by 
// Suykens, J.A.K., Van Gestel, T., De Brabanter, J., De Moor, B.,
// Vandewalle, J., Least Squares Support Vector Machines,
// World Scientific, Singapore, 2002.     (Matlab ref)

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

    fConfA, fConfB : double;

    fPrealloc : IMatrix;
    
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
    function PropTypeOfName(const Name : string) : TPropType; override;

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

    // determines the sigmoid probabilistics according to the svm output.
    procedure SigmoidTraining( cl : TSVMClassifier; var confA, confB : double);

    function PolyKernelData(augTrainSet : IMatrix) : IMatrix;
    function GaussKernelData(augTrainSet : IMatrix) : IMatrix;
    function SigmoidKernelData(augTrainSet : IMatrix) : IMatrix;
    function TrainSet(doAugment : boolean) : IMatrix;

    procedure LearnLagrangian(y : IMatrix); 
    procedure LearnLeastSquares(y : IMatrix);
    function GetMargin: double;
  protected
    // todo: add support for weighted learning
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    property Margin : double read GetMargin;  // just a property
  
    procedure AfterConstruction; override;

    procedure SetProps(const Props : TSVMProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses BaseMatrixExamples, MatrixConst, MathUtilFunc, math;

procedure TanhFunc(var value : double);
begin
     if value < -10 
     then
         value := -1
     else if value > 10 
     then
         value := 1
     else
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

function TSVMLearner.DoLearn(const weights : Array of double): TCustomClassifier;
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

    //fDists : TDoubleDynArray;
begin
     numCl := IndexOfClasses(idx, cl);
     if numCl <> 2 then
        raise ESVMClassException.Create('Error only 2 class problems are supported');

     // store indices for faster diagonal multiplication
     if fProps.learnMethod = lmLeastSquares
     then
         fZIdx := idx[0]
     else
         fZIdx := idx[0];

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
             augExmpl.SetSubMatrix(0, 0, augExmpl.Width, augExmpl.Height - 1);
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

     // ##############################################
     // #### Create confidence parameters
     SigmoidTraining(TSVMClassifier(Result), TSVMClassifier(Result).fConfA, TSVMClassifier(Result).fConfB);
end;

procedure TSVMLearner.LearnLagrangian(y : IMatrix);
var tol : double;
    maxIter : integer;
    nu  : double;
    iter : integer;
    i : integer;
    alpha : double;
    e : IMatrix;
    Q : IMatrix;
    P : IMatrix;
    u : IMatrix;
    oldu : IMatrix;
    f : IMatrix;
    cnt : integer;
    utmp : IMatrix;
    numVec : integer;

function uDiff : double;
var u1 : IMatrix;
begin
     u1 := u.Sub(oldu);
     Result := u1.ElementwiseNorm2( False );
end;
begin
     tol := 1e-5;
     maxIter := 100000;
     nu := 1/DataSet.Count;

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
     for cnt := 0 to Q.Height - 1 do
         Q[cnt, cnt] := Q[cnt, cnt] + DataSet.Count;

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

          utmp := u.ScaleAndAdd(1, alpha);

          f := Q.Mult(u);
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

     // calc D*u
     for i := 0 to Length(fZIdx) - 1 do
         u.Vec[ fZIdx[i] ] := -u.Vec[ fZIdx[i] ];

     y.UseFullMatrix;
     y.MultInPlace(u);
     faStar := y;

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

     if numVec = 0 then
        raise ESVMLearnerException.Create('No support vector found.');
     SetLength(fVectIdx, NumVec);
end;

procedure TSVMLearner.LearnLeastSquares(y: IMatrix);
var kx : IMatrix;
    grIndex : TDoubleDynArray;
    counter: Integer;
    A : IMatrix;
    b : IMatrix;
    gIdx : IMatrix;
begin
     // ensure function is symmetric
     //kx = (kx+kx')/2 + diag(1./boxconstraint);
     kx := y.Transpose;
     kx.AddInplace(y);
     kx.ScaleInPlace(1/2);

     for counter := 0 to kx.Width - 1 do
         kx[counter, counter] := kx[counter, counter] + 1;

     // create hessian
     // H =((groupIndex * groupIndex').*kx);
     gIdx := TDoubleMatrix.Create( 1, DataSet.Count, 1 );
     for counter := 0 to Length(fZIdx) - 1 do
         gIdx.Vec[fZIdx[counter]] := -1;
     gIdx.MultInPlaceT2(gIdx);
     kx.ElementWiseMultInPlace(gIdx);
     gIdx := nil;    
     
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
    dotproduct : IMatrix;
begin
     dotproduct := augTrainSet.MultT1( augTrainSet );
     if fProps.kernelType = svmPolyInhomogen then
        dotproduct.AddInplace(1);

     Result := TDoubleMatrix.Create;
     Result.Assign(dotproduct);

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

          col.ScaleAndAddInPlace(fProps.offset, fProps.scale);
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


function TSVMLearner.GetMargin: double;
begin
     Result := 0;
     if Assigned(faStar) then
        Result := 1/faStar.ElementwiseNorm2;
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
     //x := TDoubleMatrix.Create( [0.1, 0.1, 0], 3, 1 );
     // kernel evaluation
     case fProps.kernelType of
       svmPoly,
       svmPolyInhomogen : x := PolyKernel(Example);
       svmGauss: x := GaussKernel(Example);
       svmSigmoid: x := SigmoidKernel(Example);
     end;

     // weighting function (the classification plane)
     //x.TransposeInPlace;
     x.MultInPlace(fW);
     x.AddInplace(fBias);

     // classify
     confidence := Abs(x[0, 0]);
     if x[0, 0] > 0
     then
         Result := fClassVal2
     else
         Result := fClassVal1;

     // confidence defined as 1/(1 + exp(a * f(input) + b))
     // defined in Probalistic Outputs for Support Vector Machines and comparisons to regularized likelihood methods
     if (fConfA <> 0) and (fConfB <> 0) then
     begin
          confidence := fConfA*confidence + fConfB;

          // intervall check to avoid over/underflow
          if confidence > 12
          then
              confidence := 0
          else if confidence < -12
          then
              confidence := 1
          else
              confidence := 1/(1 + exp(confidence));

          confidence := 1 - confidence;
     end;
end;

function TSVMClassifier.AugmentedExample(Example: TCustomExample): IMatrix;
var x : integer;
begin
     if not Assigned(fPrealloc) then
        fPrealloc := TDoubleMatrix.Create(fSV.Width, 1, 1);
     Result := fPrealloc;

     // Create a matrix of the feature vectors -> this classifier only understands matrices
     for x := 0 to Example.FeatureVec.FeatureVecLen - 1 do
         Result[x, 0] := Example.FeatureVec[x];

     // ###########################################
     // #### Autoscaling according to the train set
     if fProps.autoScale and Assigned(fScaleMean) and Assigned(fScaleFact) then
     begin
          Result.Vec[fSV.Width - 1] := 1;
          Result.SetSubMatrix(0, 0, fScaleMean.Width, 1);
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

     fConfA := 0;
     fConfB := 0;

     fScaleMean := nil;
     fScaleFact := nil;
     if Assigned(scaleMean) then
        fScaleMean := scaleMean.Transpose;
     if Assigned(scaleFact) then
        fScaleFact := scaleFact.Transpose;
     
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
     value := value + 1;
     mul := value;
     for cnt := 0 to fProps.order - 2 do
         value := value*mul;
end;

function TSVMClassifier.PolyKernel(Example: TCustomExample): IMatrix;
var k : IMatrix;
begin
     k := AugmentedExample(Example);
     Result := k.MultT2(fSV); 

     if (fProps.order > 1) or (fProps.kernelType = svmPolyInhomogen) then
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
     
     Result := k.MultT2(fSV); 

     Result.ScaleAndAddInPlace(fProps.offset, fProps.scale);
     Result.ElementwiseFuncInPlace(TanhFunc);
end;

function TSVMClassifier.GaussKernel(Example: TCustomExample): IMatrix;
var k : IMatrix;
    gamma : double;
    i : integer;
    sum : IMatrix;
begin
     Result := TDoubleMatrix.Create(fSV.Height, 1);
     k := AugmentedExample(Example);
     gamma := -1/(2*sqr(fProps.sigma));
     sum := TDoubleMatrix.Create( k.Width, k.Height );
     for i := 0 to fSV.Height - 1 do
     begin
          sum.SetRow(0, fSV, i);
          sum.SubInPlace(k);
          Result.Vec[i] := exp(gamma*sum.ElementwiseNorm2(False));
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

function TSVMClassifier.PropTypeOfName(const Name: string): TPropType;
begin
     if (CompareText(Name, 'SVMW') = 0) or (CompareText(Name, 'SVMSV') = 0) or
        (CompareText(Name, 'SVMBIAS') = 0) or (CompareText(Name, 'SVMScaleMean') = 0) or
        (CompareText(Name, 'SVMScaleFact') = 0)
     then
         Result := ptObject
     else if (CompareText(Name, 'SVMClass1') = 0) or (CompareText(Name, 'SVMClass2') = 0) or
             (CompareText(Name, 'SVMAutoScale') = 0) or (CompareText(Name, 'SVMLearnMethod') = 0) or
             (CompareText(Name, 'SVMKernelType') = 0) or (CompareText(Name, 'SVMPolyOrder') = 0)
     then
         Result := ptInteger
     else if (CompareText(Name, 'SVMSlack') = 0) or (CompareText(Name, 'SVMGaussSigma') = 0) or
             (CompareText(Name, 'SVMSigmoidOffset') = 0) or (CompareText(Name, 'SVMSigmoidScale') = 0)
     then
         Result := ptDouble
     else
         Result := inherited PropTypeOfName(Name);
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

// appendix 5 from probailistic outputs for Support Vector machines and
// comparison to regularized likelihood methods
procedure TSVMLearner.SigmoidTraining(cl: TSVMClassifier; var confA, confB: double);
var prior0, prior1 : integer;
    svmOut : TDoubleDynArray;
    target : Array of boolean;
    counter: Integer;
    it: Integer;
    pp : TDoubleDynArray;
    count : integer;
    a, b, c, d, e : double;
    i: Integer;
    t : double;
    hiTarget : double;
    loTarget : double;
    lambda : double;
    oldErr : double;
    d1, d2 : double;
    oldA, oldB : double;
    err : double;
    det : double;
    p : double;
    l1 : double;
    l2 : double;
    diff : double;
    scale : double;
begin
     // preparation: get original
     SetLength(target, DataSet.Count);
     SetLength(svmOut, dataSet.Count);
     SetLength(pp, dataset.Count);
     prior0 := 0;
     prior1 := 0;
     for counter := 0 to dataset.Count - 1 do
     begin
          if dataSet.Example[counter].ClassVal = 1
          then
              inc(prior0)
          else
              inc(prior1);

          target[counter] := dataset.Example[counter].ClassVal = 1;
          // note: in the bare case the confidence is the distance to the plane aka svmout
          cl.Classify(dataset.Example[counter], svmOut[counter]);
     end;

     for counter := 0 to Length(pp) - 1 do
         pp[counter] := (prior1 + 1)/(prior0 + prior1 + 2);

     confA := 0;
     confB := ln((prior0 + 1)/(prior1 + 1));
     count := 0;

     hiTarget := (prior1 + 1)/(prior1 + 2);
     loTarget := 1/(prior0 + 2);
     lambda := 0.001;
     oldErr := 1e300;
     t := 0;


     for it := 1 to 100 do
     begin
          a := 0;
          b := 0;
          c := 0;
          d := 0;
          e := 0;

          // compute hessian & gradient of error function with respect to confA and confB
          for i := 0 to dataSet.Count - 1 do
          begin
               if target[i]
               then
                   t := hiTarget
               else
                   t := loTarget;

               d1 := pp[i] - t;
               d2 := pp[i]*(1 - pp[i]);
               a := a + sqr(svmOut[i])*d2;
               b := b + d2;
               c := c + svmOut[i]*d2;
               d := d + svmOut[i]*d1;
               e := e + d1;
          end;

          // if gradient is really small stop
          if (abs(d) < 1e-9) and (abs(e) < 1e-9) then
             break;

          oldA := confA;
          oldB := confB;

          err := 0;
          // loop until goodness of fit increases
          while True do
          begin
               det := (a + lambda)*(b + lambda) - sqr(c);

               if SameValue(det, 0) then
               begin
                    lambda := lambda*10;
                    continue;
               end;

               confA := oldA + ((b + lambda)*d - c*e)/det;
               confB := oldB + ((a + lambda)*e - c*d)/det;

               // now, compute the goodness of fit
               err := 0;
               for i := 0 to dataSet.Count - 1 do
               begin
                    p := svmOut[i]*confA + confB;

                    if p > 20
                    then
                        p := 0
                    else if p < -20
                    then
                        p := 1
                    else
                        p := 1/(1 + exp(p));

                    pp[i] := p;

                    // at this setp make sure log(0) returns - 200
                    if SameValue(p, 0)
                    then
                        l1 := -200
                    else
                        l1 := ln(p);

                    if SameValue(p, 1)
                    then
                        l2 := -200
                    else
                        l2 := ln(1 - p);

                    err := err - t*l1 + (1 - t)*l2;
               end;

               if err < olderr*(1 + 1e-7) then
               begin
                    lambda := lambda*0.1;
                    break;
               end;

               // error did not decrease: increase stabilizer by factor of 10 and try again
               lambda := lambda*10;

               // something went wrong -> give up
               if lambda > 1e6 then
                  break;
          end;

          diff := err - olderr;
          scale := 0.5*(err - olderr + 1);
          if (diff > -1e-3 * scale) and (diff < 1e-7 * scale)
          then
              inc(count)
          else
              count := 0;

          olderr := err;
          if count = 3 then
             break;
     end;
end;

initialization
   RegisterMathIO(TSVMClassifier);

end.
