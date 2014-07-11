unit ImportVectorMachines;

// #################################################
// #### Implementation of the Import vector machines algorithm
// #################################################

// algorithm based on: http://www.ipb.uni-bonn.de/ivm/

interface

uses BaseClassifier, Matrix, Types;

const cIVInf = $FFFFFFFF;
type
  TImportVectorLearnType = (ivSingle, ivCrossValidaton);
type
  TImportVectorMachinesProps = class(TObject)
    LearnType : TImportVectorLearnType;
    NumAddPoints : LongWord;  // maximum number of points tested for adding to the subset (default all points cIVInf)
    MaxIter : LongWord;       // maximum number of iterations (maximum number of import vectors (cIVInf takes all points)
    Epsilon : double;         // stopping criterion for convergence proof - default 0.001
    deltaK : double;          // interval for computing the ratio of the neegative loglikelihood - default 3
    Sigma : double;           // kernel parameter (default 0.2)
    Lambda : double;          // regularization parameter (e^-14)

    // params for crossvalidation
    Sigmas : TDoubleDynArray; // sigmas to be tested default sqrt(1./(2*2^(-10:1:3))
    LambdaStart : double;       // exp(10);
    LambdaEnd : double;         // exp(-5);
    Lambdas : TDoubleDynArray;

    constructor Create;
  end;

type
  TImportVectorMachine = class(TCustomClassifier)
  public
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;
  end;

// #################################################
// #### Learner class for the import vector machines classifier
type
  TImportVectorMachineLearner = class(TCustomWeightedLearner)
  private
    procedure ProbFunc(var Elem : Double);    
  private
    fProps : TImportVectorMachinesProps;
    function DiffSampleFromSet(const S : TIntegerDynArray; doRandomize : boolean; numSamples : integer) : TIntegerDynArray;
    function GreedySelect(K, KS, Kreg, y : integer; z, pts, S : IMatrix) : integer;
    procedure Exponential(var val : double);
  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    function CreateKernel(m1, m2 : IMatrix; sigma : double) : IMatrix;
    procedure SetProperties(const props : TImportVectorMachinesProps);

    destructor Destroy; override;
  end;

implementation

uses SysUtils, Math;

{ TImportVectorMachine }

function TImportVectorMachineLearner.CreateKernel(m1, m2 : IMatrix;
  sigma: double): IMatrix;
var x1: Integer;
    x2: Integer;
    value : double;
    y : Integer;
begin
     Result := TDoubleMatrix.Create(m1.Width, m2.Width);

     sigma := 1/(sqr(sigma)*2);
     for x1 := 0 to m2.Width - 1 do
     begin
          for x2 := 0 to m1.Width - 1 do
          begin
               value := 0;
               for y := 0 to m1.Height - 1 do
                   value := value + sqr(m1[x2, y] - m2[x1, y]);
               value := value*sigma;

               Result[x1, x2] := exp(-value);
          end;
     end;
end;

destructor TImportVectorMachineLearner.Destroy;
begin
     inherited;
end;

function TImportVectorMachineLearner.DiffSampleFromSet(
  const S: TIntegerDynArray; doRandomize : boolean; numSamples : integer): TIntegerDynArray;
var i, j, x, y, tmp, k : integer;
    isInS : boolean;
    len : integer;
begin
     SetLength(Result, DataSet.Count);

     k := 0;
     for i := 0 to DataSet.Count - 1 do
     begin
          isInS := False;
          for j := 0 to Length(S) - 1 do
          begin
               isInS := S[j] = i;
               if isInS then
                  break;
          end;
          if isInS then
             continue;

          Result[k] := i;
          inc(k);
     end;

     if doRandomize then
     begin
          // shuffle the array
          for i := 0 to 2*k - 1 do
          begin
               x := Random(k);
               y := Random(k);
               tmp := Result[x];
               Result[x] := Result[y];
               Result[y] := tmp;
          end;
     end;

     SetLength(Result, Min(k, numSamples));
end;

function TImportVectorMachineLearner.DoLearn(
  const weights: array of double): TCustomClassifier;
var sum : double;
    i, j : Integer;
    classLabels : TIntegerDynArray;
    classIdx : TIntIntArray;
    numClasses : integer;
    classFact : double;
    W : TDoubleMatrixDynArr;
    phi : IMatrix;
    x, y : Integer;
    numFeatures : integer;
    K : IMatrix;
    Kreg : IMatrix;
    Q : Integer;
    Probabilities : IMatrix;
    z : IMatrix;
    S : TIntegerDynArray;
    KS, KRz : IMatrix;
    alpha : IMatrix;
    tmp1, tmp2 : IMatrix;
    diagIdx : Integer;
    yMp : double;
    pts : TIntegerDynArray;
    nAdd : Cardinal;
    p : Integer;
    bestN : integer;
begin
     nAdd := fProps.NumAddPoints;
     numClasses := IndexOfClasses(classIdx, classLabels);

     // #################################################
     // #### Init probabilities
     sum := 0;
     for i := 0 to Length(weights) - 1 do
         sum := sum + weights[i];
     sum := 1/sum;
     Probabilities := TDoubleMatrix.Create(Length(weights), numClasses);

     classFact := 1/numClasses;
     for y := 0 to numClasses - 1 do
         for x := 0 to Length(weights) - 1 do
             Probabilities[x, y] := classFact*weights[x]*sum;

     SetLength(W, numClasses);
     for y := 0 to numClasses - 1 do
         W[y] := Probabilities.ElementwiseFunc(ProbFunc);

     KS := TDoubleMatrix.Create(1, Length(weights));
     alpha := TDoubleMatrix.Create(numClasses, 1);

     // compute kernel
     numFeatures := DataSet.Example[0].FeatureVec.FeatureVecLen;
     phi := TDoubleMatrix.Create(DataSet.Count, numClasses + 1);
     for x := 0 to phi.Width - 1 do
     begin
          phi[X, 0] := 1;
          for y := 0 to numFeatures - 1 do
              phi[x, y + 1] := DataSet.Example[y].FeatureVec[x];
     end;

     K := CreateKernel(phi, phi, fProps.Sigma);

     // #################################################
     // #### iterated reweighted least squares optimization with greedy selection
     z := TDoubleMatrix.Create(numClasses, DataSet.Count);
     for Q := 0 to fProps.MaxIter - 1 do
     begin
          // compute z from actual alpha
          z := KS.Mult(alpha);
          for x := 0 to numClasses - 1 do
          begin
               for diagIdx := 0 to z.Width - 1 do
               begin
                    yMp := ifthen(DataSet[diagIdx].ClassVal = classLabels[y], 1, 0);
                    yMp := yMp - Probabilities[diagIdx, y];
                    z[x, diagIdx] := z[x, diagIdx] + 1/W[y][diagIdx, 0]*yMp;
               end;
          end;
          z.ScaleInPlace(1/DataSet.Count);

          // compute KRz
          KRz := TDoubleMatrix.Create(numClasses, 1);
          KS.TransposeInPlace;
          for x := 0 to numClasses - 1 do
          begin
               z.SetSubMatrix(x, 0, 1, z.Height);
               tmp1 := KS.Mult(W[x]);
               tmp1.MultInPlace(z);

               KRz[x, 0] := tmp1[0, 0];
               FreeAndNil(tmp1);
          end;
          z.UseFullMatrix;

          // choose points to test int he subset
          if nAdd = cIVInf
          then
              pts := DiffSampleFromSet(S, False, DataSet.Count)
          else
              pts := DiffSampleFromSet(S, True, nAdd);

          // greedy selection
          bestN := GreedySelect(K, KS, Kreg, y, z, pts, S);
          for p := 0 to Length(pts) - 1 do
          begin

          end;
     end;

end;

function TImportVectorMachineLearner.DoUnweightedLearn: TCustomClassifier;
begin
     Result := TImportVectorMachine.Create;
end;

function TImportVectorMachineLearner.GreedySelect(K, KS, Kreg, y, z, pts,
  S: IMatrix): integer;
var R : IMatrix;
    cc, pp, qq : Integer;
    A, alpha : IMatrix;
    KRKT : IMatrix;
    rvec : IMatrix;
    dat : IMatrix;
    aplpha : IMatrix;
    sum_y : IMatrix;
    KTrans, KTrans1 : IMatrix;
    tmp : IMatrix;
    N : integer;
    Q : integer;
    C : Integer;
    P : integer;
    pointP : integer;
    i : integer;
begin
     N := K.Height;
     Q := KS.Height;
     C := Kreg.Height;
     P := y.Height;

     // compute probability weights  y*(1-y)
     R := TDoubleMatrix.Create(y.Width, y.Height, 1);
     R := R.Sub(Y);
     R := Y.ElementWiseMultInPlace(R);

     alpha := TDoubleMatrix.Create(Q + 1, C);
     A := TDoubleMatrix.Create(C, N);

     // test each point to be in the subset
     for pp := 0 to Length(pts) - 1 do
     begin
          sum_y := TDoubleMatrix.Create(N, 1);
          
          // set subset matrix kernel
          K.SetSubMatrix(pts[pp] - 1, 0, 1, K.Height);
          KS.AssignSubMatrix(K, Q, 0);

          // get regularization matrix
          pointP := pts[pp] - 1;
          for qq := 0 to Q - 1 do
          begin
               KReg[Q, qq] := K[pointP, Round(S[0, qq] - 1)];
               KReg[qq, Q] := K[Round(s[0, qq] - 1), pointP];
          end;

          Kreg[Q, Q] := K[pointP, pointP];

          for cc := 0 to C - 1 do
          begin
               // get product K' * R * K
               KRKT := K.Transpose;
               for i := 0 to KRKT.Width - 1 do
               begin
                    KRKT.SetSubMatrix(i, 0, 1, KTrans.Height);
                    KRKT.ScaleInPlace(R[c, i]);
               end;
               KRKT.UseFullMatrix;
               KRKT.MultInPlace(K);

               // parameters
               KTrans := KS.TransposeInPlace;
               for i := 0 to KTrans.Width - 1 do
               begin
                    KTrans.SetSubMatrix(i, 0, 1, KTrans.Height);
                    KTrans.ScaleInPlace(R[c, i]);
               end;

               z.SetSubMatrix(c, 0, 1, z.Height);
               KTrans1 := TDoubleMatrix.Create;
               KTrans1.Assign(KTrans);
               KTrans.MultInPlace(z);

               KTrans1.MultInPlace(KS);
               KTrans1.Scale(1/N);

               tmp := KS.Scale(fProps.Lambda);
               KTrans1.AddInplace(tmp);
               tmp := nil;

               kTrans1.InvertInPlace;
               KTrans1.MultInPlace(KTrans);

               alpha.UseFullMatrix;
               alpha.AssignSubMatrix(KTrans1, 0, cc);

               // new activities
               KTrans := KS.Transpose;
               Alpha.SetSubMatrix(c, 0, 1, Alpha.Height);
               KTrans := Alpha.Mult(KTrans);
               KTrans.ElementwiseFuncInPlace(Exponential);

               A.AssignSubMatrix(KTrans, c, 0);

               A.SetSubMatrix(0, c, A.Width, 1);
               sum_y.AddInplace(A);
               A.UseFullMatrix;
          end;
     end;
end;

procedure TImportVectorMachineLearner.ProbFunc(var Elem: Double);
begin
     Elem := elem*(1-elem);
end;

procedure TImportVectorMachineLearner.SetProperties(
  const props: TImportVectorMachinesProps);
begin
     fProps := Props;
end;

{ TImportVectorMachine }

function TImportVectorMachine.Classify(Example: TCustomExample;
  var confidence: double): integer;
begin
     Result := 0;
     confidence := 0;
end;

{ TImportVectorMachinesProps }

constructor TImportVectorMachinesProps.Create;
var
  i: Integer;
begin
     inherited Create;

     LearnType := ivSingle;
     NumAddPoints := cIVInf;
     MaxIter := cIVInf;
     Epsilon := 0.001;
     deltaK := 3;
     Sigma := 0.2;
     Lambda := exp(-14);

     SetLength(Sigmas, 12);
     SetLength(Lambdas, 12);
     for i := -8 to 3 do
     begin
          Sigmas[i + 8] := sqrt(1/(2*power(2, i)));
          Lambdas[i + 8] := exp(i);
     end;

     LambdaStart := exp(10);
     LambdaEnd := exp(-5);
end;

end.
