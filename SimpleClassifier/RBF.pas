// ###################################################################
// #### This file is part of the artificial intelligence project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2015, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################

unit RBF;

// ###########################################
// #### Simple radial basis function classifier
// ###########################################

interface

uses SysUtils, Classes, BaseClassifier, Matrix, Types, BaseMathPersistence;

type
  TRBFCenterType = (rbAll, rbRandom, rbClusterMean, rbClusterMedian);
  TRBFKernel = (rbGauss, rbMultQuad, rbInvMultqud, rbInvQuad);
  TRBFWeightLearning = (wlLeastSquares); //, wlBackProp);
  TRBFProperties = record
    kernel : TRBFKernel;
    centerType : TRBFCenterType;
    RBFlearnAlgorithm : TRBFWeightLearning;
    randomCenterPerc : double;
    sigma : double;  // gauss kernel param     
    epsilon : double; // multiquad and inverse mulutquad kernel - see http://en.wikipedia.org/wiki/Radial_basis_function
    augmentBase : boolean; // adds a bias term when mapping the feature vector
  end;

// ######################################################
// #### Radial basis classifier - 
// maps the input using the given kernel. The classification
// is defined by the maximum of all output elements (argmax)
type
  TRBFClassifier = class(TCustomClassifier)
  private
    fCenters : TDoubleMatrix;      // centers used to map the elements
    fW : TDoubleMatrixDynArr;      // weights of the "linear" neuron for all classes
    fClassVals : TIntegerDynArray; // original class values (e.g. 0, 1, 2,...)

    fKernelType : TRBFKernel;
    fSigma : double;              // for gaussian kernels todo: each center shall have it's own sigma
    fEpsilon : double;            // param for multiquad and inverse mulutquad kernel
    fAugmentBase : boolean;
    
    fAlpha, fBeta : double;       // shortcuts for the gauss kernel

    fIdx : Integer;               // persistence counter
    procedure AlphaFromSigma;
  private
    type
      TKernelFunc = function (exmpl : TDoubleMatrix; data : IMatrix) : double of object;
  protected
    function GaussKernel(exmpl : TDoubleMatrix; data : IMatrix) : double;
    function MultiquadKernel(exmpl : TDoubleMatrix; data : IMatrix) : double;
    function InverseMultiquadKernel(exmpl : TDoubleMatrix; data : IMatrix) : double;
    function InverseQuadKernel(exmpl : TDoubleMatrix; data : IMatrix) : double;

    function MapToRBFSpace(data : TDoubleMatrix) : TDoubleMatrix; // input: data set in matrix form - output mapped by rbf
    
    function OnLoadObject(Obj : TBaseMathPersistence) : boolean; override;
    function OnLoadObject(const Name : string; obj : TBaseMathPersistence) : boolean; override;
    procedure OnLoadDoubleProperty(const Name : String; const Value : double); override;
    procedure OnLoadIntProperty(const Name : String; Value : integer); override;
    procedure OnLoadIntArr(const Name : String; const Value : TIntegerDynArray); override;
    procedure OnLoadBeginList(const Name : String; count : integer); override;
    procedure OnLoadEndList; override;

    procedure DefineProps; override;
    function PropTypeOfName(const Name : string) : TPropType; override;

  public
    property Centers : TDoubleMatrix read fCenters;

    function Classify(Example : TCustomExample; var confidence : double) : integer; overload; override;

    constructor Create(const props : TRBFProperties; W : TDoubleMatrixDynArr; Centers : TDoubleMatrix; const clVals : TIntegerDynArray; augmentBase : boolean);
    destructor Destroy; override;
  end;

  ERBFLearnError = class(Exception);

  TRadialBasisLearner = class(TCustomWeightedLearner)
  private
    fProps : TRBFProperties;

    fclIdx : TIntIntArray;
    fclassVals : TIntegerDynArray;

    fCenters : TDoubleMatrix;
    fWeights : TDoubleMatrix;
    fW : TDoubleMatrixDynArr;
    fCenterClassVals : TIntegerDynArray;

    fData : TDoubleMatrix;

    procedure RBFCenterAll;
    procedure RBFCenterRandom;
    procedure RBFCenterMean;
    procedure RBFCenterMedian;
  protected
    // weighting is achieved by:
    // -> weighting the output response in case all elements are in the data set
    // -> perform a weighted mean
    // -> (nothing on median)
    // -> perform a "preferred" random sort with some ticketing system
    //    a high weight will get you a higher number of tickets thus the probability
    //    to be in the final randomized dataset is higher
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure SetProps(const props : TRBFProperties);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;

    destructor Destroy; override;
  end;

implementation

uses BaseMatrixExamples, MatrixConst, Math, MathUtilFunc;

// ###########################################
// #### RBF Classifier
// ###########################################

constructor TRBFClassifier.Create(const props: TRBFProperties;
  W: TDoubleMatrixDynArr; Centers: TDoubleMatrix;
  const clVals: TIntegerDynArray; augmentBase : boolean);
begin
     fW := W;
     fCenters := Centers;
     fClassVals := clVals;
     fAugmentBase := augmentBase;

     fKernelType := props.kernel;
     case fKernelType of
       rbGauss: fSigma := props.sigma;
       rbMultQuad: fEpsilon := props.epsilon;
       rbInvMultqud: fEpsilon := props.epsilon;
       rbInvQuad: fEpsilon := props.epsilon;
     end;

     AlphaFromSigma;

     inherited Create;
end;

destructor TRBFClassifier.Destroy;
var counter: Integer;
begin
     fCenters.Free;

     for counter := 0 to Length(fW) - 1 do
         fW[counter].Free;

     inherited;
end;

// ###########################################
// #### Kernel functions
// ###########################################

function TRBFClassifier.GaussKernel(exmpl : TDoubleMatrix; data : IMatrix) : double;
var n2 : double;
begin
     data.Assign(exmpl, True);
     data.SubInPlace(fCenters);
     n2 := data.ElementwiseNorm2;

     Result := fAlpha*exp(-n2*fBeta);
end;

function TRBFClassifier.InverseMultiquadKernel(exmpl : TDoubleMatrix; data : IMatrix) : double;
begin
     Result := 1/MultiquadKernel(exmpl, data);
end;

function TRBFClassifier.InverseQuadKernel(exmpl : TDoubleMatrix; data : IMatrix) : double;
begin
     Result := Sqr(1/MultiquadKernel(exmpl, data));
end;

function TRBFClassifier.MultiquadKernel(exmpl : TDoubleMatrix; data : IMatrix) : double;
var n2 : double;
begin
     data.Assign(exmpl, true);
     data.SubInPlace(fCenters);
     data.ScaleInPlace(fEpsilon);
     n2 := data.ElementwiseNorm2;
     Result := sqrt(1 + n2);
end;

procedure TRBFClassifier.AlphaFromSigma;
begin
     fAlpha := 0;
     fBeta := 0;

     if fsigma <> 0 then
     begin
          fAlpha := 1/(sqrt(2*pi)*fsigma);
          fBeta := 1/(2*sqr(fsigma));
     end;
end;

function TRBFClassifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var maxVal : double;
    counter: Integer;
    projData : IMatrix;
    actVal : IMatrix;
    maxIdx : integer;
    exmplMtx : IMatrix;
begin
     confidence := 0;

     exmplMtx := TDoubleMatrix.Create(1, Example.FeatureVec.FeatureVecLen);
     for counter := 0 to exmplMtx.Height - 1 do
         exmplMtx.Vec[counter] := Example.FeatureVec[counter];

     // ###########################################
     // #### The neuron with the highest output wins:
     maxVal := -MaxDouble;
     maxIdx := 0;

     projData := MapToRBFSpace(exmplMtx.GetObjRef);
     projData.TransposeInPlace;

     for counter := 0 to Length(fW) - 1 do
     begin
          actVal := fW[counter].Mult(projData);

          if actVal[0, 0] > maxVal then
          begin
               maxIdx := counter;
               maxVal := actVal[0, 0];
          end;
     end;

     Result := fClassVals[maxIdx];

     confidence := Max(0, Min(1, maxVal));
end;

function TRBFClassifier.MapToRBFSpace(data : TDoubleMatrix) : TDoubleMatrix;
var kernelFun : TKernelFunc;
    x : Integer;
    exmplIdx : integer;
    help : IMatrix;
begin
     case fKernelType of
       rbGauss:      kernelFun := GaussKernel;
       rbMultQuad:   kernelFun := MultiquadKernel;
       rbInvMultqud: kernelFun := InverseMultiquadKernel;
       rbInvQuad:    kernelFun := InverseQuadKernel;
     else
         raise Exception.Create('Not yet implemented');
     end;

     // ###########################################
     // #### Apply kernel on all examples
     if faugmentBase
     then
         Result := TDoubleMatrix.Create(fCenters.Width + 1, Data.Width) // including the bias!
     else
         Result := TDoubleMatrix.Create(fCenters.Width, Data.Width);

     help := TDoubleMatrix.Create;

     // go through examples
     for exmplIdx := 0 to data.Width - 1 do
     begin
          data.SetSubMatrix(exmplIdx, 0, 1, data.Height);

          // centers (aka neuron)
          for x := 0 to fCenters.Width - 1 do
          begin
               fCenters.SetSubMatrix(x, 0, 1, fCenters.Height);

               Result[x, exmplIdx] := kernelFun(data, help)
          end;

          fCenters.UseFullMatrix;
     end;

     if fAugmentBase then
     begin
          // handle bias -> augment matrix
          Result.SetSubMatrix(fCenters.Width, 0, 1, Result.Height);
          Result.SetValue(1);
     end;

     // ###########################################
     // #### undo sub matrix selection
     fCenters.UseFullMatrix;
     Data.UseFullMatrix;
     Result.UseFullMatrix;
end;

// ###########################################
// #### Persistence
// ###########################################

const cClassLabels = 'labels';
      cSigma = 'sigma';
      cEpsilon = 'epsilon';
      cKernel = 'kernel';
      cCenter = 'centers';
      cWeights = 'weights';
      cAugmentBase = 'augmentbase';

procedure TRBFClassifier.OnLoadBeginList(const Name: String; count: integer);
begin
     fIdx := -1;
     if SameText(Name, cWeights) then
     begin
          SetLength(fW, count);
          fIdx := 0;
     end
     else
         inherited;
end;

procedure TRBFClassifier.OnLoadDoubleProperty(const Name: String;
  const Value: double);
begin
     if SameText(Name, cSigma) then
     begin
          fSigma := Value;
          if fSigma <> 0 then
          begin
               fAlpha := 1/(sqrt(2*pi)*fsigma);
               fBeta := 1/(2*sqr(fsigma));
          end;
     end
     else if SameText(Name, cEpsilon)
     then
         fEpsilon := Value
     else
         inherited;
end;

procedure TRBFClassifier.OnLoadEndList;
begin
     fIdx := -1;
end;

procedure TRBFClassifier.OnLoadIntArr(const Name: String;
  const Value: TIntegerDynArray);
begin
     if SameText(Name, cClassLabels)
     then
         fClassVals := Value
     else
         inherited;
end;

procedure TRBFClassifier.OnLoadIntProperty(const Name: String; Value: integer);
begin
     if SameText(Name, cAugmentBase)
     then
         fAugmentBase := Value = 1
     else if SameText(Name, cKernel)
     then
         fKernelType := TRBFKernel(Value)
     else
         inherited;
end;

function TRBFClassifier.OnLoadObject(const Name: string;
  obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if SameText(Name, cCenter)
     then
         fCenters := obj as TDoubleMatrix
     else
         Result := inherited OnLoadObject(Name, obj);
end;

function TRBFClassifier.OnLoadObject(Obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if fIdx >= 0 then
     begin
          fW[fIdx] := obj as TDoubleMatrix;
          inc(fIdx);
     end
     else
         Result := inherited OnLoadObject(obj);
end;

procedure TRBFClassifier.DefineProps;
var counter: Integer;
begin
     inherited;

     AddIntArr(cClassLabels, fClassVals);
     AddDoubleProperty(cSigma, fSigma);
     AddDoubleProperty(cEpsilon, fEpsilon);

     AddIntProperty(cKernel, Integer(fKernelType));
     AddIntProperty(cAugmentBase, Integer(fAugmentBase));

     AddObject(cCenter, fCenters);
     BeginList(cWeights, Length(fW));
     for counter := 0 to Length(fW) - 1 do
         AddObject(fW[counter]);
     EndList;
end;

function TRBFClassifier.PropTypeOfName(const Name: string): TPropType;
begin
     if (CompareText(Name, cClassLabels) = 0) or (CompareText(Name, cKernel) = 0) or
        (CompareText(Name, cAugmentBase) = 0)
     then
         Result := ptInteger
     else if (CompareText(Name, cSigma) = 0) or (CompareText(Name, cEpsilon) = 0)
     then
         Result := ptDouble
     else if (CompareText(Name, cCenter) = 0) or (CompareText(Name, cWeights) = 0)
     then
         Result := ptObject
     else
         Result := inherited PropTypeOfName(Name);

end;


// ###########################################
// #### Learning algorithm
// ###########################################

{ TRadialBasisLearner }

class function TRadialBasisLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TRBFClassifier;
end;

destructor TRadialBasisLearner.Destroy;
begin
     inherited;
end;

function TRadialBasisLearner.DoLearn(const weights : Array of double) : TCustomClassifier;
var phi : IMatrix;
    counter: Integer;
    y : Integer;
    x : Integer;
    yC : TDoubleMatrix;
    maxWeight : double;
begin
     fWeights := TDoubleMatrix.Create;
     fWeights.Assign( weights, 1, Length(weights));
     maxWeight := fWeights.Max;
     fWeights.ScaleInPlace(1/maxWeight);

     IndexOfClasses(fclIdx, fclassVals);

     // we need the dataset as matrix -> convert
     if DataSet is TMatrixLearnerExampleList
     then
         fData := TMatrixLearnerExampleList(DataSet).Matrix
     else
     begin
          fData := TDoubleMatrix.Create(DataSet.Count, DataSet.Example[0].FeatureVec.FeatureVecLen);

          for x := 0 to DataSet.Count - 1 do
          begin
               for y := 0 to DataSet[x].FeatureVec.FeatureVecLen - 1 do
                   fData[y, x] := DataSet[x].FeatureVec[y];
          end;
     end;

     try
        // ###########################################
        // #### Calculate centers
        case fProps.centerType of
          rbAll: RBFCenterAll;
          rbRandom: RBFCenterRandom;
          rbClusterMean: RBFCenterMean;
          rbClusterMedian: RBFCenterMedian;
        end;

        // we need a copy since the kernel function uses submatrices on both params
        if DataSet is TMatrixLearnerExampleList
        then
            fData := fCenters.Clone
        else
            fdata.Assign(fCenters);

        // ###########################################
        // #### Calculate coefficient matrix
        SetLength(fW, Length(fclassVals));

        Result := TRBFClassifier.Create(fProps, fW, fCenters, fclassVals, fProps.augmentBase);
        try
           if fProps.RBFlearnAlgorithm = wlLeastSquares then
           begin
                // calculate a weight matrix for each given class
                phi := TRBFClassifier(Result).MapToRBFSpace(fData);

                if phi.PseudoInversionInPlace <> srOk then
                   raise ERBFLearnError.Create('Error could not compute phi');

                yC := TDoubleMatrix.Create(1, fCenters.Width);

                if fProps.centerType = rbAll then
                begin
                     // restrict neuron output in case of weighting!
                     for counter := 0 to Length(fW) - 1 do
                     begin
                          for y := 0 to fCenters.Width - 1 do
                          begin
                               if fCenterClassVals[y] = fclassVals[counter]
                               then
                                   yC.Vec[y] := 1*fweights.Vec[y]
                               else
                                   yC.Vec[y] := -1*fweights.Vec[y];
                          end;

                          fW[counter] := phi.Mult(yC);
                          fW[counter].TransposeInPlace;
                     end;
                end
                else
                begin
                     for counter := 0 to Length(fW) - 1 do
                     begin
                          for y := 0 to fCenters.Width - 1 do
                          begin
                               if fCenterClassVals[y] = fclassVals[counter]
                               then
                                   yC.Vec[y] := 1
                               else
                                   yC.Vec[y] := -1;
                          end;

                          fW[counter] := phi.Mult(yC);
                          fW[counter].TransposeInPlace;
                     end;
                end;

                yC.Free;
           end
           else
               raise Exception.Create('Not yet implemented');
        except
              Result.Free;
              raise;
        end;
     finally
            fWeights.Free;
            fData.Free;
     end;
end;

procedure TRadialBasisLearner.RBFCenterRandom;
var sortIdx : TIntegerDynArray;
    counter, i : integer;
    randCnt : integer;
    idx, idx1 : integer;
    itemIdx : integer;
    tmp : integer;
    numItems : integer;
    mtx : TDoubleMatrix;
    y : integer;
    buckets : TDoubleDynArray;
    sumBuckets, bucketVal : double;
    maxBucketVal : double;
    doBucketSearch : boolean;
// returns false if all buckets are the same size
function InitBuckets(const clIdx : Array of integer) : boolean;
var weightMax, weightMin : double;
    sumWeight : double;
    counter : integer;
    aWeight : double;
    sumBucket : double;
    mult : double;
begin
     Result := False;

     weightMax := 0;
     weightMin := 1;
     sumWeight := 0;
     for counter := 0 to High(clIdx) do
     begin
          aWeight := fWeights.Vec[clIdx[counter]];
          sumWeight := sumWeight + aWeight;
          weightMax := Max(weightMax, aWeight);
          weightMin := Min(weightMin, aWeight);
     end;
     SetLength(buckets, Length(clIdx));

     if SameValue(weightMax, weightMin, 1e-3)
     then
         exit
     else
         mult := 1/(weightMax - weightMin);

     Result := True;
     weightMin := weightMin - 1e-3;
     sumBucket := 0;
     for counter := 0 to Length(buckets) - 1 do
     begin
          buckets[counter] := (fWeights.Vec[clIdx[counter]] - weightMin)*mult;
          sumBucket := sumBucket + buckets[counter];
     end;

     // normalize sum to 1
     mult := 1/sumBucket;
     for counter := 0 to Length(buckets) - 1 do
         buckets[counter] := buckets[counter]*mult;
end;
begin
     mtx := nil;

     if DataSet is TMatrixLearnerExampleList then
        mtx := TMatrixLearnerExampleList(DataSet).Matrix;

     numItems := 0;
     for counter := 0 to Length(fclIdx) - 1 do
         numItems := numItems + Max(1, Round(Length(fclIdx[counter])*fProps.randomCenterPerc));

     fCenters := TDoubleMatrix.Create(numItems, DataSet.Example[0].FeatureVec.FeatureVecLen);
     SetLength(fCenterClassVals, numItems);

     itemIdx := 0;
     for counter := 0 to Length(fclIdx) - 1 do
     begin
          // ###########################################
          // #### randomize the class index list, so we can take
          // the first randomCenterPerc*Length items out of it in a later step.

          // take weighting into account and allow the weighting algorithm to
          // change random probabilities.
          sortIdx := Copy(fclIdx[counter], 0, Length(fclIdx[counter]));
          doBucketSearch := InitBuckets(sortIdx);
          maxBucketVal := 1;

          for randCnt := Length(sortIdx) - 1 downto 0 do
          begin
               // search in different size buckets:
               bucketVal := Dataset.Rand.Random*maxBucketVal;  // -> 0 - 1

               if doBucketSearch then
               begin
                    // find bucket idx in which the random value falls
                    sumBuckets := 0;
                    idx1 := randCnt;
                    for i := 0 to randCnt do
                    begin
                         sumBuckets := sumBuckets + buckets[i];
                         if (bucketVal <= sumBuckets) then
                         begin
                              // found:
                              idx1 := i;

                              // update buckets
                              bucketVal := buckets[i];
                              buckets[i] := buckets[randCnt];
                              buckets[randCnt] := bucketVal;
                              maxBucketVal := maxBucketVal - bucketVal;
                              break;
                         end;
                    end;
               end
               else
                   idx1 := DataSet.Rand.RandInt(randCnt + 1);

               tmp := sortIdx[idx1];
               sortIdx[idx1] := sortIdx[randCnt];
               sortIdx[randCnt] := tmp;
          end;

          // ###########################################
          // #### Now get the center
          for idx := 0 to Max(1, Round(Length(fclIdx[counter])*fProps.randomCenterPerc)) - 1 do
          begin
               if Assigned(mtx)
               then
                   fCenters.SetColumn(itemIdx, mtx, sortIdx[idx])
               else
               begin
                    for y := 0 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
                        fCenters[itemIdx, y] := DataSet[ sortIdx[idx] ].FeatureVec[y];
               end;

               fCenterClassVals[itemIdx] := fclassVals[counter];

               inc(itemIdx);
          end;
     end;
end;

procedure TRadialBasisLearner.RBFCenterAll;
var x, y : integer;
begin
     SetLength(fCenterClassVals, DataSet.Count);
     if DataSet is TMatrixLearnerExampleList then
     begin
          fCenters := TDoubleMatrix.Create;
          fCenters.Assign(TMatrixLearnerExampleList(DataSet).Matrix);
     end
     else
     begin
          // Create a matrix of the feature vectors
          fCenters := TDoubleMatrix.Create(DataSet.Count, DataSet[0].FeatureVec.FeatureVecLen);
          for y := 0 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
              for x := 0 to DataSet.Count - 1 do
                  fCenters[x, y] := DataSet[x].FeatureVec[y];
     end;

     for y := 0 to Length(fCenterClassVals) - 1 do
         fCenterClassVals[y] := DataSet[y].ClassVal;
end;

procedure TRadialBasisLearner.RBFCenterMean;
var x, y : integer;
    mtx : TDoubleMatrix;
    clIdx : integer;
    weightVal : double;
    vec : TDoubleMatrix;
    aWeight : double;
begin
     // take the mean of all elements belonging to one class -> use that as center
     fCenters := TDoubleMatrix.Create(Length(fclassVals), DataSet[0].FeatureVec.FeatureVecLen);
     SetLength(fCenterClassVals, Length(fclassVals));
     mtx := nil;

     if DataSet is TMatrixLearnerExampleList then
        mtx := TMatrixLearnerExampleList(DataSet).Matrix;


     vec := TDoubleMatrix.Create(1, fCenters.Height);
     for x := 0 to fCenters.Width - 1 do
     begin
          fCenters.SetSubMatrix(x, 0, 1, fCenters.Height);
          weightVal := 0;

          for clIdx := 0 to Length(fclIdx[x]) - 1 do
          begin
               aWeight := fWeights.Vec[fClIdx[x][clIdx]];
               if Assigned(mtx) then
               begin
                    mtx.SetSubMatrix( fClIdx[x][clIdx], 0, 1, mtx.Height );

                    vec.SetColumn(0, mtx, fClIdx[x][clIdx]);
                    vec.ScaleInPlace( aWeight );
                    fCenters.AddInplace(vec);
               end
               else
               begin
                    for y := 0 to fCenters.Height - 1 do
                        vec.Vec[y] := aWeight*DataSet[x].FeatureVec[y];

                    fCenters.AddInplace(vec);
               end;

               weightVal := weightVal + aWeight;
          end;

          fCenterClassVals[x] := fclassVals[x];
          fCenters.ScaleInPlace(1/weightVal);
     end;

     mtx.UseFullMatrix;
     fCenters.UseFullMatrix;
end;

procedure TRadialBasisLearner.RBFCenterMedian;
var x, y : integer;
    col : TDoubleDynArray;
    clIdx : integer;
    featureLen : integer;
    numExmpl : integer;
begin
     featureLen := DataSet[0].FeatureVec.FeatureVecLen;
     // take the mean of all elements belonging to one class -> use that as center
     fCenters := TDoubleMatrix.Create(Length(fclassVals), featureLen);
     SetLength(fCenterClassVals, Length(fclassVals));

     for clIdx := 0 to Length(fclIdx) - 1 do
     begin
          numExmpl := Length(fclIdx[clIdx]);
          SetLength(col, Length(fclIdx[clIdx]));

          for y := 0 to featureLen - 1 do
          begin
               // extract the features for the selected class and the selected feature index
               for x := 0 to Length(fclIdx[clIdx]) - 1 do
                   col[x] := DataSet[fclIdx[clIdx][x]].FeatureVec[y];

               if numExmpl and 1 = 0
               then
                   fCenters[clIdx, y] := (KthLargest(col, numExmpl div 2) + KthLargest(col, numExmpl div 2 + 1))/2
               else
                   fCenters[clIdx, y] := KthLargest(col, numExmpl div 2);
          end;

          fCenterClassVals[clIdx] := fclassVals[clIdx];
     end;
end;

procedure TRadialBasisLearner.SetProps(const props: TRBFProperties);
begin
     fProps := props;
end;

initialization
  RegisterMathIO(TRBFClassifier);

end.
