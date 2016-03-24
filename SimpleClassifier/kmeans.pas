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

unit kmeans;

interface

uses SysUtils, Classes, Types, BaseClassifier, Matrix, BaseMathPersistence;

type
  TKMeansClusterInit = (ciRandom, ciKmeansPlusPlus);

type
  TKMeansProps = record
    numCluster : integer;
    searchPerClass : boolean;   // if true then numClusters are searched per class
                                // if flase then numClusters are searched on the complete data set
                                // (this defines then the number of output classes) - it's then like
                                // unsupervised learning
    useMedian : boolean;        // kmeans median algorithm (manhattan distance and median update) 
    MaxNumIter : integer;       // maximum number of iterations until convergence
    centChangePerc : double;    // algorithm stops if center change is lower than the given percentage (0 - 1)!
    initCluster : TKMeansClusterInit;
  end;

// ###########################################
// #### KMeans classifier:
// searches for the closest eukledian distance to the centers -> closest center wins
type
  TKMeans = class(TCustomClassifier)
  private
    fCenters : IMatrix;
    fClassVals : TIntegerDynArray;
  protected
    function OnLoadObject(const Name: string; obj: TBaseMathPersistence): boolean; override;
    procedure OnLoadIntArr(const Name : String; const Value : TIntegerDynArray); override;
    function ClassifyWithMtx(mtx : TDoubleMatrix; var confidence : double) : integer;

    procedure DefineProps; override;
  public
    property Centers : IMatrix read fCenters;
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;

    constructor Create(centers : IMatrix; const classVals : TIntegerDynArray);
  end;

// ###########################################
// #### KMeans learner:
// -> iterative adjustment of class centers until convergence (or maxiterations) is reached
type
  TKMeansLearner = class(TCustomWeightedLearner)
  private
   fProps : TKMeansProps;
   fclIdx : TIntIntArray;
   fclassVals : TIntegerDynArray;
   fData : TDoubleMatrix;

   function MatrixDataSet(out ownsObj : boolean) : TDoubleMatrix;
   function RandomClassCenters(var centerClassVals : TIntegerDynArray) : IMatrix;
   function KMeanPPClassCenters(var centerClassVals : TIntegerDynArray) : IMatrix;
  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure SetProps(const props : TKMeansProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;

    constructor Create;
    destructor Destroy; override;
  end;

implementation

uses Math, BaseMatrixExamples;

{ TKMeans }

function TKMeans.ClassifyWithMtx(mtx: TDoubleMatrix;
  var confidence: double): integer;
var clIdx : integer;
    counter : integer;
    minDist : Double;
    actDist : IMatrix;
    dist : double;
begin
     clIdx := 0;
     minDist := MaxDouble;
     
     // ###########################################
     // #### Find closest center
     for counter := 0 to fCenters.Width - 1 do
     begin
          fCenters.SetSubMatrix(counter, 0, 1, fCenters.Height);
          
          actDist := fCenters.Sub(mtx);
          dist := actDist.ElementwiseNorm2;

          if dist < minDist then
          begin
               clIdx := counter;
               minDist := dist;
          end;
     end;

     fCenters.UseFullMatrix;

     Result := fClassVals[clIdx];
     confidence := 0;   
end;

function TKMeans.Classify(Example: TCustomExample;
  var confidence: double): integer;
var counter : integer;
    exmplMtx : TDoubleMatrix;
begin
     exmplMtx := TDoubleMatrix.Create(1, Example.FeatureVec.FeatureVecLen);
     try
        for counter := 0 to exmplMtx.Height - 1 do
            exmplMtx.Vec[counter] := Example.FeatureVec[counter];
         
        Result := ClassifyWithMtx(exmplMtx, confidence);  
     finally
            exmplMtx.Free;
     end;
end;

constructor TKMeans.Create(centers: IMatrix; const classVals: TIntegerDynArray);
begin
     inherited Create;

     fCenters := centers;
     fClassVals := classVals;
end;


// ###########################################
// #### Persistence
// ###########################################

const cClassLabels = 'labels';
      cCenter = 'centers';


procedure TKMeans.DefineProps;
begin
     inherited;

     AddIntArr(cClassLabels, fClassVals);
     AddObject(cCenter, fCenters.GetObjRef);
end;

procedure TKMeans.OnLoadIntArr(const Name: String;
  const Value: TIntegerDynArray);
begin
     if SameText(Name, cClassLabels) 
     then
         fClassVals := Value
     else
         inherited;
end;

function TKMeans.OnLoadObject(const Name: string; obj: TBaseMathPersistence): boolean;
begin
     Result := True;
     if SameText(Name, cCenter) 
     then
         fCenters := obj as TDoubleMatrix
     else 
         Result := inherited OnLoadObject(Name, obj);
end;


{ TKMeansLearner }

class function TKMeansLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TKMeans;
end;

constructor TKMeansLearner.Create;
begin
     fProps.numCluster := 2;
     fProps.searchPerClass := False;
     fProps.MaxNumIter := 100;
     fProps.initCluster := ciRandom;
     fProps.centChangePerc := 0.01; // 1% change = convergence
     fProps.useMedian := False;

     inherited Create;
end;

destructor TKMeansLearner.Destroy;
begin
     inherited;
end;

procedure TKMeansLearner.SetProps(const props: TKMeansProps);
begin
     fProps := Props;
end;

function TKMeansLearner.DoLearn(const weights: array of double): TCustomClassifier;
var centers : IMatrix;
    newCenters : IMatrix;
    centClasses : TIntegerDynArray;
    x, y : integer;
    iter : integer;
    ownsObj : boolean;
    numItems : integer;
    centIdx : integer;
    minDist : double;
    exmplCnt: Integer;
    centCnt : integer;
    actVal : IMatrix;
    actDist : double;
    numCentvals : TIntegerDynArray;
    centerIndexes : Array of TIntegerDynArray;
    centerChange : double;
    curCenterChange : double;
    weight : double;
    classCnt: Integer;
    delCenters : Array of Boolean;
    counter: Integer;
    exmplClass : Integer;
    exmplErr : TIntIntArray;
    conf : double;
    i: Integer;
    clIdx: Integer;
    maxIdx, maxVal : integer;
    sumWeight : double;
    srtData : TDoubleDynArray;
    srtIdx : TIntegerDynArray;
    sumIdx : integer;
begin
     // weights are used in this algorithm only in the mean calculation where the centers move!

     // ###########################################
     // #### Initialize data
     IndexOfClasses(fclIdx, fclassVals);

     // in case we don't have the per cluster property set:
     if not fProps.searchPerClass then
     begin
          numItems := 0;
          for iter := 0 to Length(fClIdx) - 1 do
              numItems := numItems + Length(fClIdx[iter]);

          x := Length(fClIdx[0]);
          SetLength(fclIdx[0], numItems);
          for iter := 1 to Length(fClIdx) - 1 do
          begin
               Move(fClIdx[iter][0], fClIdx[0][x], Length(fclIdx[iter])*sizeof(integer));
               inc(x, Length(fclIdx[iter]));
          end;
          
          SetLength(fclIdx, 1);
     end;
     
     
     fData := MatrixDataSet(ownsObj);
     try
        // ###########################################
        // #### Initialize centers
        case fProps.initCluster of
          ciRandom: centers := RandomClassCenters(centClasses);
          ciKmeansPlusPlus: centers := KMeanPPClassCenters(centClasses);
        else
            raise Exception.Create('Not yet implemented');
        end;

        SetLength(delCenters, centers.Width);
        newCenters := TDoubleMatrix.Create(centers.Width, centers.Height);
        SetLength(numCentvals, centers.width);

        centerChange := MaxDouble;
        SetLength(centerIndexes, centers.Width, fData.Width);

        // ###########################################
        // #### iterate until class centers converge
        for iter := 0 to fProps.MaxNumIter - 1 do
        begin
             // initialize new centers 
             newCenters.SetValue(0);
             FillChar(numCentVals[0], sizeof(integer)*Length(numCentvals), 0);
              
             // do it for each class (or only for one if the learning is unsupervised) 
             for classCnt := 0 to Length(fClIdx) - 1 do
             begin
                  // find the closest center for each example
                  for exmplCnt := 0 to Length(fClIdx[classCnt]) - 1 do
                  begin
                       minDist := maxDouble;
                       centIdx := -1;

                       fData.SetSubMatrix(fClIdx[classCnt][exmplCnt], 0, 1, fData.Height);
                       for centCnt := 0 to fProps.numCluster - 1 do
                       begin
                            centers.SetSubMatrix(classCnt*fProps.numCluster + centCnt, 0, 1, centers.Height);
                            actVal := fData.Sub(centers);

                            if fProps.useMedian then
                            begin
                                 // use sum of absolute values (manhattan distance)
                                 actval.AbsInPlace;
                                 actVal.SumInPlace(False);
                                 actDist := actVal.Vec[0];
                            end
                            else
                                actDist := actVal.ElementwiseNorm2;

                            if actDist < minDist then
                            begin
                                 centIdx := classCnt*fProps.numCluster + centCnt;
                                 minDist := actDist;
                            end;
                       end;

                       // add the example to the new center
                       centerIndexes[centIdx][numCentVals[centIdx]] := fClIdx[classCnt][exmplCnt];
                       inc(numCentvals[centIdx]);
                  
                       centers.UseFullMatrix;
                  end;
             end;

             fData.UseFullMatrix;
             
             // ###########################################
             // #### Recalculate centers (weighted)
             if fProps.useMedian then
             begin
                  // calculate median of each center. Todo: introduce weighted median!
                  for centcnt := 0 to centers.Width - 1 do
                  begin
                       actVal := TDoubleMatrix.Create(numCentVals[centcnt], newCenters.Height);

                       weight := 0;
                       for exmplCnt := 0 to numCentvals[centCnt] - 1 do
                       begin
                            actVal.SetColumn(exmplCnt, fData, CenterIndexes[centcnt][exmplCnt]);
                            weight := weight + weights[centerIndexes[centcnt][exmplCnt]];
                       end;

                       delCenters[centCnt] := numCentvals[centCnt] = 0;
                       if numCentVals[centCnt] > 0 then
                       begin
                            // weighted median: sort -> then increase index until cummulative weights are 0.5*weight
                            // todo: eventually move the routine to the matrix class
                            for y := 0 to fData.Height - 1 do
                            begin
                                 actVal.SetSubMatrix(0, y, actVal.Width, 1);
                                 srtData := actVal.SubMatrix;
                                 SetLength(srtIdx, Length(srtData));
                                 for x := 0 to Length(srtIdx) - 1 do
                                     srtIdx[x] := x;
                                 IdxQuickSort(srtData, srtIdx, 0, Length(srtData) - 1);
                                 
                                 // median: 
                                 weight := 0.5*weight;
                                 sumWeight := 0;
                                 sumIdx := -1;
                                 for x := 0 to Length(srtData) - 1 do
                                 begin
                                      inc(sumIdx);
                                      sumWeight := sumWeight + weights[ srtIdx[x] ];
                                      if sumWeight >= weight then
                                         break;
                                 end;

                                 newCenters[centCnt, y] := srtData[ srtIdx[sumIdx] ];
                            end;
                            
                           // actVal.MedianInPlace(True);
                           // newCenters.SetColumn(centCnt, actVal);
                       end;
                  end;
             end
             else
             begin
                  // standard (but weighted) kmeans
                  actVal := TDoubleMatrix.Create(1, fData.Height); 
                  for centcnt := 0 to centers.Width - 1 do
                  begin
                       newCenters.SetSubMatrix(centcnt, 0, 1, newCenters.Height);
                  
                       weight := 0;

                       for exmplCnt := 0 to numCentVals[centCnt] - 1 do
                       begin
                            actVal.SetColumn(0, fData, centerIndexes[centcnt][exmplCnt]);
                            weight := weight + weights[centerIndexes[centcnt][exmplCnt]];
                            actVal.ScaleInPlace(weights[centerIndexes[centcnt][exmplCnt]]);

                            newCenters.AddInplace(actVal);
                       end;
             
                       delCenters[centCnt] := weight = 0;
                       if weight = 0 then
                          weight := 1;
                       if weight <> 0 then
                          newCenters.ScaleInPlace(1/weight);
                  end;
             end;
             
             // calculate change and assign new centers
             newCenters.UseFullMatrix;
             centers.UseFullMatrix;

             centers.SubInPlace(newCenters);
             curCenterChange := centers.ElementwiseNorm2;

             centers.Assign(newCenters);
             
             // ###########################################
             // #### check for convergence 
             if curCenterChange*(1 + fProps.centChangePerc) >= centerChange then
                break;

             centerChange := curCenterChange;
        end;
     finally
            if ownsObj then
               fData.Free;
     end;

     // ###########################################
     // #### Create classifer
     Result := TKMeans.Create(centers, centClasses);

     if not fProps.searchPerClass then
     begin
          // optimize the class labels for minimum error 

          // classify each example and count to which class it would
          // belong -> the maximum wins

          // centClasses is shared with TKmeans 
          SetLength(exmplErr, Length(centClasses));
          for counter := 0 to Length(centClasses) - 1 do
          begin
               centclasses[counter] := counter;
               SetLength(exmplErr[counter], Length(fclassVals));
               FillChar(exmplErr[counter][0], sizeof(exmplErr[counter][0])*Length(fclassVals), 0);
          end;

          for counter := 0 to fData.Width - 1 do
          begin
               fData.SetSubMatrix(counter, 0, 1, fData.Height);
               exmplClass := TKMeans(Result).ClassifyWithMtx(fData, conf);

               for i := 0 to Length(fclassVals) - 1 do
               begin
                    if DataSet.Example[counter].ClassVal = fclassVals[i] then
                    begin
                         inc(exmplErr[exmplClass][i]);
                         break;
                    end;
               end;
          end;
          fData.UseFullMatrix;
          
          // now assign classes -> maximum wins and defines the class
          for counter := 0 to Length(centClasses) - 1 do
          begin
               maxIdx := 0;
               maxVal := exmplErr[counter][0];
               for clIdx := 1 to Length(exmplErr[counter]) - 1 do
               begin
                    if maxVal < exmplErr[counter][clIdx] then
                    begin
                         maxIdx := clIdx;
                         maxVal := exmplErr[counter][clIdx];
                    end;
               end;

               centClasses[counter] := fclassVals[maxIdx];
          end;
     end;
end;

function TKMeansLearner.MatrixDataSet(out ownsObj: boolean): TDoubleMatrix;
var x, y : integer;
begin
     // we need the dataset as matrix -> convert eventual 
     if DataSet is TMatrixLearnerExampleList 
     then
         Result := TMatrixLearnerExampleList(DataSet).Matrix
     else
     begin
          Result := TDoubleMatrix.Create(DataSet.Count, DataSet.Example[0].FeatureVec.FeatureVecLen);

          for x := 0 to DataSet.Count - 1 do
          begin
               for y := 0 to DataSet[x].FeatureVec.FeatureVecLen - 1 do
                   Result[y, x] := DataSet[x].FeatureVec[y];
          end;
     end;

     ownsObj := not (DataSet is TMatrixLearnerExampleList);
end;

function TKMeansLearner.KMeanPPClassCenters(
  var centerClassVals: TIntegerDynArray): IMatrix;
var counter : integer;
    centIdx : integer;
    exmplIdx : integer;
    centIndex : TIntegerDynArray;
    distSortIdx : TIntegerDynArray;
    distances : TDoubleDynArray;
    i : integer;
    j: Integer;
    dist : IMatrix;
    len : integer;
function IsCenterUsed(curidx : integer; aktRandVal : integer) : boolean;
var i: Integer;
begin
     Result := False;

     for i := 0 to curIdx - 1 do
         Result := Result or (centIndex[i] = aktRandVal);
end;
begin
     SetLength(centerClassVals, Length(fclIdx)*fProps.numCluster);
     SetLength(centIndex, fProps.numCluster);

     Result := TDoubleMatrix.Create(Length(centerClassVals), fData.Height);

     // rules for class centers:
     // 1.) first one is random
     // 2.) calculate the distance of each point ot the class center
     // 3.) random init next center but with probability invers to the distance
     
     for counter := 0 to Length(fclIdx) - 1 do
     begin
          SetLength(distSortIdx, Length(fclIdx[counter]));
          SetLength(distances, Length(fclIdx[counter]));
          for centIdx := 0 to fProps.numCluster - 1 do
          begin
               // distance sort
               if centIdx > 0 then
               begin
                    for i := 0 to Length(fClIdx[counter]) - 1 do
                    begin
                         distSortIdx[i] := i;
                         distances[i] := 1;
                         fData.SetSubMatrix(fClIdx[counter][i], 0, 1, fData.Height);
                         
                         for j := 0 to centIdx - 1 do
                         begin
                              Result.SetSubMatrix(counter*fProps.numCluster + j, 0, 1, Result.Height);
                              
                              dist := Result.Sub(fData);
                              distances[i] := distances[i]*dist.ElementwiseNorm2;
                         end;
                    end;

                    // sort according to the distances
                    IdxQuickSort(distances, distSortIdx, 0, Length(distSortIdx) - 1);

                    Result.UseFullMatrix;
                    fData.UseFullMatrix;
               end
               else
               begin
                    for i := 0 to Length(fclIdx[counter]) - 1 do
                        distSortIdx[i] := i;
               end;

               len := Length(fClIdx[counter]) - 1;
               repeat             
                     // use gaussian distributed random to achive higher probability on
                     // far away centers
                     exmplIdx := len - Min( Abs( Round(Length(fclIdx[counter])*RandG(0, 0.33)) ), len); 
                     exmplIdx := distSortIdx[ exmplIdx ];
               until Not IsCenterUsed(centIdx, exmplIdx);

               centIndex[centIdx] := exmplIdx;

               Result.SetColumn(centIdx + counter*fProps.numCluster, fData, fClIdx[counter][exmplIdx]);
               centerClassVals[centIdx + counter*fProps.numCluster] := fclassVals[counter];
          end;
     end;
      
    // if not fProps.searchPerClass then
//     begin
//          for counter := 0 to Length(centerClassVals) - 1 do
//              centerClassVals[counter] := fclassVals[counter] 
//     end;
end;

function TKMeansLearner.RandomClassCenters(
  var centerClassVals: TIntegerDynArray): IMatrix;
var counter : integer;
    centIdx : integer;
    exmplIdx : integer;
    centIndex : TIntegerDynArray;
function IsCenterUsed(curidx : integer; aktRandVal : integer) : boolean;
var i: Integer;
begin
     Result := False;

     for i := 0 to curIdx - 1 do
         Result := Result or (centIndex[i] = aktRandVal);
end;
begin
     SetLength(centerClassVals, Length(fclIdx)*fProps.numCluster);
     SetLength(centIndex, fProps.numCluster);

     Result := TDoubleMatrix.Create(Length(centerClassVals), fData.Height);

     for counter := 0 to Length(fclIdx) - 1 do
     begin
          for centIdx := 0 to fProps.numCluster - 1 do
          begin
               repeat               
                     exmplIdx := random(Length(fclIdx[counter]));
               until Not IsCenterUsed(centIdx, exmplIdx);

               centIndex[centIdx] := exmplIdx;

               Result.SetColumn(centIdx + counter*fProps.numCluster, fData, fClIdx[counter][exmplIdx]);
               centerClassVals[centIdx + counter*fProps.numCluster] := fclassVals[counter];
          end;
     end;
end;

initialization
  RegisterMathIO(TKMeans);

end.
