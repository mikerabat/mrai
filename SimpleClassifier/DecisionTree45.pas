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

unit DecisionTree45;

// #############################################################
// #### Implementation of a simple
// #### the C4.5 decision tree with weighted examples
// #############################################################
// ###########################################
// #### Many thanks to Normand Péladeau who greatly helped to improve the routines 
// ###########################################

interface

uses SysUtils, Classes, BaseClassifier, TreeStructs, Types, BaseMathPersistence;

type
  T45TreeLearnType = (ltFull, ltMaxDepth, ltMinSizeBranch, ltPrune);

type
  TC45Props = record
    allowFeatureMultiUse : boolean;     // allows a feature to be used in more than one node
    useGainRatio : boolean;             // use the gain ration G(S,B)/P(S,B) instaed of G(S,B) alone
    threadedGainCalc : boolean;         // parallelize the gain calculation
    numCores : integer;                 // use that man cores (0 = use the number of system cores) - you cannot use more than the number of system cores
    minNumSamplesPerThread : integer;   // number of items at least used per thread. If zero 250 is used. 
    useCountSort : boolean;             // use counting sort instead of quicksort - can only be used if the feature values are integers
    
    case LearnType : T45TreeLearnType of
      ltMaxDepth: (MaxDepth : integer);
      ltMinSizeBranch: (MinNumDataSetSize : integer);
      ltPrune: ( UseValidationSet : boolean; ValidationsetSize : double);
  end;

type
  T45NodeData = class(TObject)
  public
    NumClasses : integer;
    Classes : TIntegerDynArray;
    NumExamples : integer;
    Conf : double;               // we define the confidence as the weighted sum of the leaf elements 
                                 // to the overall weighted sum leading to the leaf
                                 // -> is 1 if all elements are correctly classified in the learning set
                                 // -> 0 < conf < 1 if the leaf is not correctly classifying due to max depth or pruning
    
    NumLeft, NumRight : integer;
    WeightedSums : TDoubleDynArray;
    FeatureSplitIndex : integer;
    SplitVal : double;

    procedure WriteToStream(Stream : TStream);
    procedure LoadFromStream(stream : TStream);

    procedure Clear;

    constructor Create;
  end;

type
  TC45Classifier = class(TCustomClassifier)
  private
    fTree : TCustomTreeItem;
    fClasses : TIntegerDynArray;

    procedure SaveTreeToStream(stream : TStream);
    procedure LoadTreeFromStream(stream : TStream);
  public
    property Tree : TCustomTreeItem read fTree;
    
    function Classify(Example : TCustomExample; var confidence : double) : integer; override;

    procedure OnLoadBinaryProperty(const Name : String; const Value; size : integer); override;
    procedure DefineProps; override;
    function PropTypeOfName(const Name : string) : TPropType; override;

    constructor Create(Tree : TCustomTreeItem);
    destructor Destroy; override;
  end;

// ###########################################################
// #### Learning C4.5 decission tree:
// http://www.cis.temple.edu/~ingargio/cis587/readings/id3-c45.html
// http://ai.stanford.edu/~ronnyk/treesHB.pdf
type
  TC45Learner = class(TCustomWeightedLearner)
  private
    fProps : TC45Props;
    fLastFeatureIdx : integer;
    fUsedFeatures : TBooleanDynArray;
    fsumRight : TDoubleDynArray;
    fClasses : TIntegerDynArray;
    fNumClasses : integer;

    fFeatureVecLen : integer;
    fCurIdx : integer;
    
    function GetNextFeatureIdx( var idx : integer ) : boolean;
    
    procedure InitializeSums(const Weights : Array of double; const dataSetIdx : TIntegerDynArray);
    procedure PruneTree(tree : TCustomTreeItem; var dataSetIdx : TIntegerDynArray);
    function SplitByProperty(const Weights : Array of double; var dataSetIdx : TIntegerDynArray; curDepth : integer) : TCustomTreeItem;
    function CalculateGain(const Weights : Array of double; const dataSetIdx : TIntegerDynArray; FeatureIdx : integer;
                           var SplitVal : double; setEntropy : double; numClasses : integer; var numElements : integer) : double;
  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure SetProperties(const Props : TC45Props);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses Math, MathUtilFunc, MtxThreadPool, Windows;

const eps = 1e-6;
      cMinNumExamplesPerThread = 250;
      c45ClassesProps = 'classes';
      c45TreeProps = 'treedata';

// ###########################################
// #### 
// ###########################################
      
procedure T45NodeData.Clear;
begin
     NumClasses := 0;
     Classes := nil;
     WeightedSums := nil;
end;

constructor T45NodeData.Create;
begin
     NumClasses := 0;
     Classes := nil;
     WeightedSums := nil;
     FeatureSplitIndex := -1;
     SplitVal := 0;
end;

procedure T45NodeData.LoadFromStream(stream: TStream);
var len : integer;
begin
     Stream.ReadBuffer(NumClasses, sizeof(integer));
     Stream.ReadBuffer(len, sizeof(integer));
     if len > 0 then
     begin
          SetLength(classes, len);
          Stream.ReadBuffer(classes[0], len*sizeof(integer));
     end;
     Stream.ReadBuffer(NumExamples, sizeof(integer));
     Stream.ReadBuffer(len, sizeof(integer));
     if len > 0 then
     begin
          SetLength(WeightedSums, len);
          Stream.WriteBuffer(WeightedSums[0], len*sizeof(double));
     end;
     Stream.ReadBuffer(FeatureSplitIndex, sizeof(integer));
     Stream.ReadBuffer(SplitVal, sizeof(double));
end;

procedure T45NodeData.WriteToStream(Stream: TStream);
var len : integer;
begin
     Stream.WriteBuffer(NumClasses, sizeof(integer));
     len := Length(Classes);
     Stream.WriteBuffer(len, sizeof(integer));
     if len > 0 then
        Stream.WriteBuffer(classes[0], len*sizeof(integer));
     Stream.WriteBuffer(NumExamples, sizeof(integer));
     len := Length(WeightedSums);
     Stream.WriteBuffer(len, sizeof(integer));
     if len > 0 then
        Stream.WriteBuffer(WeightedSums[0], len*sizeof(double));
     Stream.WriteBuffer(FeatureSplitIndex, sizeof(integer));
     Stream.WriteBuffer(SplitVal, sizeof(double));
end;

{ T45DecisionTreeLearner }

function TC45Learner.CalculateGain(const Weights: array of double;
  const dataSetIdx: TIntegerDynArray; FeatureIdx: integer;
  var SplitVal: double; setEntropy : double; numClasses : integer; var numElements : integer): double;
var i : integer;
    k : Integer;
    sortIdx : TIntegerDynArray;
    sumWeight : double;
    sumLeft : TDoubleDynArray;
    sumRight : TDoubleDynArray;
    maxEntropy : double;
    partitionEntropy : double;
    sumWeightLeft : Double;
    entropyGain : double;
    entropyL, entropyR : double;
    gs : double;
    splitEntropy : double;
    elem : TCustomLearnerExample;
    next : TCustomLearnerExample;
    mapIdx : integer;
begin
     // calculatin of the information gain IG(Ex, a) = H(Ex) - H(Ex|a)
     // where H(Ex) is the entropy of the complete set and H(Ex|a) the entropy
     // of the examples
     SplitVal := 0;

     if fProps.useCountSort 
     then
         sortIdx := CountSortIdx(dataSetIdx, FeatureIdx)
     else
         sortIdx := CalcSortIdx(dataSetIdx, FeatureIdx);

     sumWeight := 0;
     for i := 0 to Length(dataSetIdx) - 1 do
         sumWeight := sumWeight + weights[dataSetIdx[i]];

     // find entry which minimizes the entropy gain if splitted by current attribute
     SetLength(sumLeft, numClasses);
     SetLength(sumRight, numClasses);

     for i := 0 to numClasses - 1 do
     begin
          sumLeft[i] := 0;
          sumRight[i] := fSumright[i];
     end;

     // initialize the sums
     maxEntropy := -MaxDouble;
     SplitVal := 0;
     numElements := 0;

     // now go through all splits and update the "left" side ( aka all elements that belong to the left node if splitted)
     next := DataSet[DataSetIdx[sortIdx[0]]];
     for i := 0 to Length(dataSetIdx) - 2 do
     begin
          // just precalc the indices and access to the elements
          mapIdx := DataSetIdx[sortIdx[i]]; 
          elem := next;
          next := DataSet[DataSetIdx[sortIdx[i + 1]]];

          // move current element to the "left" part of the tree
          for k := 0 to numClasses - 1 do
          begin
               if elem.ClassVal = fClasses[k] then
               begin
                    sumLeft[k] := sumLeft[k] + weights[mapIdx];
                    break;
               end;
          end;

          if elem.FeatureVec[FeatureIdx] = next.FeatureVec[FeatureIdx] then
             continue;

          sumWeightLeft := 0; 
          for k := 0 to numClasses - 1 do
          begin
               sumWeightLeft := sumWeightLeft + sumLeft[k];
               sumRight[k] := fsumRight[k] - sumLeft[k];
          end;
               
          // calc overall entropy for both datasets
          
          // G(S, B) = I(S) - sum(i=1:t) abs(Si)/abs(S)*I(Si)
          // I(Si) = - sum(j=1:x)*RF(Cj, S)*ln( RF(Cj, S )
          // where Cj is the class and RF the relative frequency of examples in this class and this subset
          entropyL := 0;
          entropyR := 0;
          for k := 0 to numClasses - 1 do
          begin
               if sumLeft[k] > eps then
                  entropyL := entropyL - sumLeft[k]/sumWeightLeft*log2(sumLeft[k]/sumWeightLeft);
               if sumRight[k] > eps then
                  entropyR := entropyR - sumRight[k]/(sumWeight - sumWeightLeft)*log2(sumRight[k]/(sumWeight - sumWeightLeft));
          end;
          
          // G(S)
          splitEntropy := entropyL*sumweightLeft/sumWeight + entropyR*(sumWeight - sumWeightLeft)/sumWeight;
          gs := setEntropy - splitEntropy;
          entropyGain := 0;
          
          if fProps.useGainRatio then
          begin
               // P(S,B) = - sum(i = 1:t) abs(Si)/abs(S)*ln( abs(Si)/abs(S) )
               partitionEntropy := 0;

               // left
               if sumWeightLeft > 0 then
                  partitionEntropy := - sumWeightLeft/sumWeight*log2(sumWeightLeft/sumWeight);
               // right
               if sumWeight > sumWeightLeft then
                  partitionEntropy := partitionEntropy - (sumWeight - sumWeightLeft)/sumWeight*log2((sumWeight - sumWeightLeft)/sumWeight);

               // B
               if partitionEntropy > eps then
                  entropyGain := gs/partitionEntropy; 
          end
          else
              entropyGain := gs;

          if (entropyGain > maxEntropy) then
          begin
               numElements := i + 1;
               maxEntropy := entropyGain; 
               SplitVal := (elem.FeatureVec[FeatureIdx] + next.FeatureVec[FeatureIdx]) / 2;
          end;
     end;

     Result := maxEntropy;
end;

class function TC45Learner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = TC45Classifier;
end;

function TC45Learner.DoLearn(
  const weights: array of double): TCustomClassifier;
var tree : TCustomTreeItem;
    i : integer;
    trainingIdx, validationIdx : TIntegerDynArray;
    trainLen : integer;
begin
     SetLength(trainingIdx, Length(weights));
     SetLength(fUsedFeatures, DataSet.Example[0].FeatureVec.FeatureVecLen);
     for i := 0 to Length(fUsedFeatures) - 1 do
         fUsedFeatures[i] := False;

     for i := 0 to Length(weights) - 1 do
         trainingIdx[i] := i;

     if (fProps.LearnType = ltPrune) and (fProps.UseValidationSet) and (fProps.ValidationsetSize > 0) then
     begin
          // Create shuffle set
          trainingIdx := DataSet.Shuffle;
          trainLen := Max(1, Round(Length(weights)*(1 - fProps.ValidationsetSize)));
          SetLength(validationIdx, Length(weights) - trainLen);
          for i := trainLen to Length(weights) - 1 do
              validationIdx[i - trainLen] := trainingIdx[i];
          SetLength(trainingIdx, trainLen);
     end
     else
         validationIdx := trainingIdx;

     fLastFeatureIdx := -1;
     fFeatureVecLen := DataSet[0].FeatureVec.FeatureVecLen;
     
     tree := SplitByProperty(weights, trainingIdx, 0);

     if fProps.LearnType = ltPrune then
        PruneTree(tree, validationIdx);

     Result := TC45Classifier.Create(tree);
end;

function TC45Learner.GetNextFeatureIdx(var idx: integer): boolean;
begin
     idx := InterlockedIncrement( fCurIdx );
     Result := idx < fFeatureVecLen;
end;

procedure TC45Learner.InitializeSums(const Weights : Array of double; const dataSetIdx : TIntegerDynArray);
var i, k : integer;
begin
     SetLength(fsumRight, fNumClasses);
     SetLength(fclasses, fNumClasses);

     for k := 0 to fNumClasses - 1 do
     begin
          fSumright[k] := 0;
          fclasses[k] := -MaxInt;
     end;
     
     for i := 0 to Length(dataSetIdx) - 1 do
     begin
          for k := 0 to fnumClasses - 1 do
          begin
               if DataSet[DataSetIdx[i]].ClassVal = fclasses[k] then
               begin
                    fsumRight[k] := fsumRight[k] + Weights[dataSetIdx[i]];
                    break;
               end
               else if fClasses[k] = -MaxInt then
               begin
                    fsumRight[k] := Weights[dataSetIdx[i]];
                    fclasses[k] := DataSet[DataSetIdx[i]].ClassVal;
                    break;
               end;
          end;
     end;
end;

// pruning see: http://ai.stanford.edu/~ronnyk/treesHB.pdf
// base idea used here: reduced error pruning
// -> if a node can be pruned without raising the error replace it
procedure TC45Learner.PruneTree(tree: TCustomTreeItem; var dataSetIdx : TIntegerDynArray);
var rootNode : TTreeNode;
    rootCoorect : integer;
    numClassItems : integer;
    i : Integer;
    nodeData : T45NodeData;
    splitData : TIntegerDynArray;
    leave : TTreeLeave;
procedure SplitDataSet(var dataSetIdx, splitIdx : TIntegerDynArray);
var numItems : integer;
    i : integer;
    data : T45NodeData;
    numDataIdx : integer;
begin
     data := T45NodeData(rootNode.TreeData);
     SetLength(splitIdx, Length(dataSetIdx));
     numItems := 0;
     numDataIdx := 0;

     for i := 0 to Length(dataSetIdx) - 1 do
     begin
          if (DataSet[dataSetIdx[i]].FeatureVec[data.FeatureSplitIndex] < data.SplitVal) then
          begin
               splitIdx[numItems] := dataSetIdx[i];
               inc(numItems);
          end
          else
          begin
               dataSetIdx[numDataIdx] := dataSetIdx[i];
               inc(numDataIdx);
          end;
     end;

     SetLength(splitIdx, numItems);
     SetLength(dataSetIdx, numDataIdx);
end;

function TreeClassify(node : TCustomTreeItem; idx : integer) : integer;
var nodeData : T45NodeData;
begin
     while Assigned(node) and not (node is TTreeLeave) do
     begin
          nodeData := T45NodeData(node.TreeData);

          if DataSet[idx].FeatureVec[nodeData.FeatureSplitIndex] < nodeData.SplitVal
          then
              node := TTreeNode(node).LeftItem
          else
              node := TTreeNode(node).RightItem;

          assert(Assigned(node), 'No leave assigned');
     end;

     nodeData := T45NodeData(node.TreeData);
     Result := nodeData.Classes[0];
end;

function calcError(tree : TTreeNode; dataSetIdx : TIntegerDynArray) : integer;
var i : integer;
    numCorrect : integer;
begin
     numCorrect := 0;
     for i := 0 to Length(dataSetIdx) - 1 do
         if TreeClassify(Tree, dataSetIdx[i]) = DataSet[dataSetIdx[i]].ClassVal then
            inc(numCorrect);

     Result := numCorrect;
end;
var cl1, cl2 : integer;
begin
     // recursive implementation of the pruning algorithm
     if (tree is TTreeLeave) then
        exit;

     // ##################################################
     // #### Prune tree according to the current node
     rootNode := tree as TTreeNode;
     rootCoorect := calcError(rootNode, dataSetIdx);

     numClassItems := 0;
     cl1 := DataSet[dataSetIdx[0]].ClassVal;
     cl2 := -cl1;
     for i := 0 to Length(dataSetIdx) - 1 do
         if DataSet[dataSetIdx[i]].ClassVal = DataSet[dataSetIdx[0]].ClassVal
         then
             inc(numClassItems)
         else
             cl2 := DataSet[dataSetIdx[i]].ClassVal;

     SplitDataSet(dataSetIdx, splitData);

     if rootCoorect >= Max(Length(dataSetIdx) - numClassItems, numClassItems) then
     begin
          if Length(splitData) > 0 then
             PruneTree(rootNode.LeftItem, splitData);
          if Length(dataSetIdx) > 0 then
             PruneTree(rootNode.RightItem, dataSetIdx);
     end
     else
     begin
          // merge the datasets again
          SetLength(dataSetIdx, Length(dataSetIdx) + Length(splitData));
          if Length(splitData) > 0 then
             Move(splitData[0], dataSetIdx[Length(dataSetIdx) - Length(splitData)], Length(splitData)*sizeof(splitData[0]));

          // exchange the current node by the leave with the class with the most occurances
          // substitute root node by a leave
          if Assigned(rootNode.Parent) then
          begin
               leave := TTreeLeave.Create;
               nodeData := T45NodeData.Create;
               nodeData.SplitVal := 0;
               nodeData.FeatureSplitIndex := -1;
               SetLength(nodeData.Classes, 1);
               nodeData.Classes[0] := ifthen(Length(dataSetIdx) - numClassItems < numClassItems, cl1, cl2);
               nodeData.Conf := numClassItems/Length(dataSetIdx);
               if nodeData.Classes[0] = cl2 then
                  nodeData.Conf := 1 - nodeData.Conf;
               
               leave.TreeData := nodeData;

               if TTreeNode(rootNode.Parent).LeftItem = rootNode then
               begin
                    TTreeNode(rootNode.Parent).LeftItem.Free;
                    TTreeNode(rootNode.Parent).LeftItem := leave;
               end
               else
               begin
                    TTreeNode(rootNode.Parent).RightItem.Free;
                    TTreeNode(rootNode.Parent).RightItem := leave;
               end;
          end;
     end;
end;

procedure TC45Learner.SetProperties(
  const Props: TC45Props);
begin
     fProps := Props;
     if (fProps.numCores > numCPUCores) or (fProps.numCores = 0) then
        fProps.numCores := numCPUCores;
end;

// ###########################################
// #### Parallel processing structs
// ###########################################
type
  TConstDoubleArr = Array[0..MaxInt div sizeof(double) - 1] of double;
  PConstDoubleArr = ^TConstDoubleArr;
  TAsyncC45Rec = record
    weights : PConstDoubleArr;
    numWeights : integer;
    obj : TC45Learner;
    dataSetIdx : TIntegerDynArray;
    setEntropy : double;
    
    splitVals : TDoubleDynArray;
    gain : TDoubleDynArray;
    numElements : TIntegerDynArray;
  end;
  PAsyncC45Rec = ^TAsyncC45Rec;

procedure GainProc( rec : Pointer );
var idx : integer;
    pC45 : PAsyncC45Rec;
begin
     pC45 := PAsyncC45Rec(rec);
     while pC45^.obj.GetNextFeatureIdx( idx ) do
     begin
          if (idx = pC45^.obj.fLastFeatureIdx) or (not pC45^.obj.fProps.allowFeatureMultiUse and pC45^.obj.fUsedFeatures[idx]) then
             continue; 
          
          pC45^.gain[idx] := pC45^.obj.CalculateGain( Slice( pC45^.weights^, pC45^.numWeights ), 
                             pC45^.dataSetIdx, idx, pC45^.splitVals[idx], pC45^.setEntropy, 
                             pC45^.obj.fNumClasses, pC45^.numElements[idx] );
     end;
end;

// ###########################################
// #### Splitting 
// ###########################################

function TC45Learner.SplitByProperty(const Weights : Array of double; var dataSetIdx : TIntegerDynArray; curDepth : integer): TCustomTreeItem;
var i, j : integer;
    WeightedNumElements : TDoubleDynArray;
    NumExamples : TIntegerDynArray;
    overallWeight : double;
    clFound : boolean;
    clVal : integer;
    entropy : double;
    prob : double;
    maxGain : double;
    splitVals : TDoubleDynArray;
    maxSplitVal : double;
    gains : TDoubleDynArray;
    maxSplitFeatureIdx : integer;
    maxnumElements : integer;
    nodeData : T45NodeData;
    numElements : TIntegerDynArray;
    leftData : TIntegerDynArray;
    rightData : TIntegerDynArray;
    leftIdx, rightIdx : integer;
    tempD : double;
    tempI : integer;
    overallNumExamples : integer;
    asynRec : TAsyncC45Rec;
    numObj : integer;
    calls : IMtxAsyncCallGroup;
    minNumSamplesPerThread : integer;

function SelectLeaveByWeights : TTreeLeave;
var i : integer;
    sumWeights : double;
begin
     // assign a leave node to the class with the best performance
     nodeData := T45NodeData.Create;
     nodeData.NumClasses := 1;
     nodeData.Classes := Copy(fClasses, 0, fNumClasses);
     nodeData.WeightedSums := Copy(WeightedNumElements, 0, fNumClasses);
     nodeData.FeatureSplitIndex := -1;
     nodeData.SplitVal := 0;
     nodeData.NumExamples := overallNumExamples;
     nodeData.NumLeft := 0;
     nodedata.NumRight := 0;
     sumWeights := nodeData.WeightedSums[0];

     for i := 1 to fnumClasses - 1 do
     begin
          sumWeights := sumWeights + nodeData.WeightedSums[i];
          if nodeData.WeightedSums[i] > nodeData.WeightedSums[0] then
          begin
               tempi := nodeData.Classes[i];
               nodeData.Classes[i] := nodeData.Classes[0];
               nodeData.Classes[0] := tempi;

               tempD := nodeData.WeightedSums[i];
               nodeData.WeightedSums[i] := nodeData.WeightedSums[0];
               nodeData.WeightedSums[0] := tempD;
          end;
     end;
     nodeData.Conf := 0;
     if sumWeights > eps then
        nodeData.Conf := nodeData.WeightedSums[0]/sumWeights;

     Result := TTreeLeave.Create;
     Result.TreeData := nodeData;
end;
     
begin
     assert(Length(dataSetIdx) > 0, 'Error empty dataset for splitting');

     minNumSamplesPerThread := cMinNumExamplesPerThread;
     if fProps.minNumSamplesPerThread > 0 then
        minNumSamplesPerThread := fProps.minNumSamplesPerThread;
        
     
     // find the property with the highest information gain -> split there.
     SetLength(WeightedNumElements, 2);
     SetLength(NumExamples, 2);
     SetLength(fClasses, 2);
     fnumClasses := 0;
     overallWeight := 0;
     overallNumExamples := 0;

     // calculate statistics
     for i := 0 to Length(dataSetIdx) - 1 do
     begin
          clVal := DataSet[dataSetIdx[i]].ClassVal;
          clFound := False;

          for j := 0 to fnumClasses - 1 do
          begin
               if fClasses[j] = clVal then
               begin
                    clFound := True;
                    WeightedNumElements[j] := weightedNumElements[j] + weights[dataSetIdx[i]];
                    inc(NumExamples[j]);
                    break;
               end;
          end;

          if not clFound then
          begin
               if Length(fClasses) <= fnumClasses + 1 then
               begin
                    SetLength(fClasses, Min(Length(fclasses) + 1024, 2*fnumClasses));
                    SetLength(weightedNumElements, Length(fClasses));
                    SetLength(NumExamples, Length(fClasses));
               end;

               fClasses[fNumClasses] := clVal;
               weightedNumElements[fNumClasses] := weights[dataSetIdx[i]];
               NumExamples[fNumClasses] := 1;
               inc(fnumClasses);
          end;

          overallWeight := overallWeight + weights[dataSetIdx[i]];
     end;

     for i := 0 to fnumClasses - 1 do
         inc(overallNumExamples, NumExamples[i]);

     // check the max depth property:
     if ( (fProps.LearnType = ltMaxDepth) and (curDepth = fProps.MaxDepth) ) or 
        ( (fProps.LearnType = ltMinSizeBranch) and (Length(dataSetIdx) < fProps.MinNumDataSetSize) ) then
     begin
          Result := SelectLeaveByWeights;
          
          exit;
     end;

     // ###########################################################
     // #### split with the normal procedure
     if fnumClasses = 1 then
     begin
          nodeData := T45NodeData.Create;
          nodeData.NumClasses := 1;
          nodeData.Classes := Copy(fClasses, 0, fnumClasses);
          nodeData.FeatureSplitIndex := -1;
          nodeData.SplitVal := 0;
          nodeData.NumExamples := overallNumExamples;
          nodeData.NumLeft := overallNumExamples;
          nodeData.NumRight := overallNumExamples;

          // we are finished and ended up in a leave...
          Result := TTreeLeave.Create;
          Result.TreeData := nodeData;
     end
     else
     begin
          // ###########################################################
          // #### Split the dataset:

          // calculate overall entropy or "impurity" of that node
          entropy := 0;
          for i := 0 to fnumClasses - 1 do
          begin
               prob := weightedNumElements[i]/(overallWeight + eps);
               entropy := entropy - prob*Log2(prob);
          end;

          InitializeSums( weights, dataSetIdx );
          
          // calculate information gain if one would split the node at this property
          maxGain := -MaxDouble;
          maxSplitVal := 0;
          maxSplitFeatureIdx := 0;
          numElements := nil;
          maxnumElements := 0;

          SetLength(gains, fFeatureVecLen);
          SetLength(splitVals, fFeatureVecLen);
          SetLength(numElements, fFeatureVecLen);
          
          for i := 0 to Length(gains) - 1 do
              gains[i] := -MaxDouble;

          if fProps.threadedGainCalc and (Length(dataSetIdx) > 50) and (fFeatureVecLen > 3) then
          begin
               // prepare the threads
               fCurIdx := -1;
               numObj := Min(fProps.numCores, fFeatureVecLen);
               numObj := Min(numObj, Max(1, Length(dataSetIdx) div minNumSamplesPerThread));

               asynRec.weights := @weights[0];
               asynRec.numWeights := Length(Weights);
               asynRec.obj := self;
               asynRec.dataSetIdx := dataSetIdx;
               asynRec.setEntropy := entropy;
               asynRec.splitVals := splitVals;
               asynRec.gain := gains;
               asynRec.numElements := numElements;

               calls := nil;
               if numObj > 1 then
               begin
                    calls := MtxInitTaskGroup;

                    for i := 0 to numObj - 2 do
                        calls.AddTaskRec(@GainProc, @asynRec);
               end;

               // execute the last task in the current thread               
               GainProc( @asynRec );

               if calls <> nil then
                  calls.SyncAll;
               calls := nil;
          end
          else
          begin
               for i := 0 to fFeatureVecLen - 1 do
               begin
                    if (i = fLastFeatureIdx) or (not fProps.allowFeatureMultiUse and fUsedFeatures[i]) then
                       continue;
               
                    gains[i] := CalculateGain(Weights, dataSetIdx, i, splitVals[i], entropy, fnumClasses, numElements[i]);
               end;
          end;


          for i := 0 to Length(gains) - 1 do
          begin
               if (gains[i] > maxGain) or ((gains[i] = maxGain) and (i <> fLastFeatureIdx)) then
               begin
                    maxSplitFeatureIdx := i;
                    maxSplitVal := splitVals[i];
                    maxGain := gains[i];
                    maxnumElements := numElements[i];
               end;
          end;

          // create the nodes
          SetLength(leftData, maxnumElements);
          SetLength(rightData, Length(dataSetIdx) - maxnumElements);

          leftIdx := 0;
          rightIdx := 0;

          // split data
          for i := 0 to Length(dataSetIdx) - 1 do
          begin
               if DataSet[dataSetIdx[i]].FeatureVec[maxSplitFeatureIdx] < maxSplitVal then
               begin
                    // resize if necessary -> e.g. there are elements doubled
                    if leftIdx >= Length(leftData) then
                       SetLength(leftData, Length(dataSetIdx));
                    leftData[leftIdx] := dataSetIdx[i];
                    inc(leftIdx);
               end
               else
               begin
                    if rightIdx >= Length(rightData) then
                       SetLength(rightData, Length(dataSetIdx));
                    rightData[rightIdx] := dataSetIdx[i];
                    inc(rightIdx);
               end;
          end;

          SetLength(rightData, rightIdx);
          SetLength(leftData, leftIdx);

          // release memory, we don't want to use O(n^2) memory consumption (max O(2n)
          dataSetIdx := nil;

          // ###########################################
          // #### Check if there is a gain at all (or we can split at least one element)
          // if not then create a leave
          if (maxGain = -MaxDouble) or (leftData = nil) or (rightData = nil) then
          begin
               // get the class with the most elements -> this is then the result
               Result := SelectLeaveByWeights; 
               exit;
          end
          else
          begin
               // ##########################################################
               // #### Create result
               fLastFeatureIdx := maxSplitFeatureIdx;
               fUsedFeatures[maxSplitFeatureIdx] := True;

               Result := TTreeNode.Create;
               
               nodeData := T45NodeData.Create;
               nodeData.NumClasses := fnumClasses;
               nodeData.Classes := Copy(fClasses, 1, fNumClasses);
               nodeData.WeightedSums := Copy(WeightedNumElements, 1, fNumClasses);
               nodeData.FeatureSplitIndex := maxSplitFeatureIdx;
               nodeData.SplitVal := maxSplitVal;
               nodeData.NumExamples := overallNumExamples;
               nodeData.NumLeft := leftIdx;
               nodeData.NumRight := rightIdx;
               
               TTreeNode(Result).LeftItem := SplitByProperty(weights, leftData, curDepth + 1);
               TTreeNode(Result).RightItem := SplitByProperty(weights, rightData, curDepth + 1);

               Result.TreeData := nodeData;
          end;
     end;
end;

{ T45DecisionTree }

function TC45Classifier.Classify(Example: TCustomExample;
  var confidence: double): integer;
var node : TCustomTreeItem;
    nodeData : T45NodeData;
    num : integer;
begin
     // todo: incorporate missing value detection and confidence calculation
     node := fTree;
     confidence := 1;
     while Assigned(node) and not (node is TTreeLeave) do
     begin
          nodeData := T45NodeData(node.TreeData);
          
          if Example.FeatureVec[nodeData.FeatureSplitIndex] < nodeData.SplitVal 
          then
              node := TTreeNode(node).LeftItem
          else
              node := TTreeNode(node).RightItem;

          assert(Assigned(node), 'No leave assigned');
     end;

     assert(node is TTreeLeave, 'Error no leave found');
     nodeData := T45NodeData(node.TreeData);
     Result := nodeData.Classes[0];
     confidence := nodeData.Conf;  // just what we learned at the training
end;

constructor TC45Classifier.Create(Tree: TCustomTreeItem);
var dataObj : T45NodeData;
begin
     inherited Create;

     dataObj := T45NodeData(Tree.TreeData);
     fClasses := dataObj.Classes;
     fTree := Tree;
end;

procedure TC45Classifier.DefineProps;
var mem : TMemoryStream;
begin
     if Length(fClasses) > 0 then
        AddBinaryProperty(c45ClassesProps, fClasses[0], Length(fClasses)*sizeof(longInt));

     if Assigned(fTree) then
     begin
          mem := TMemoryStream.Create;
          try
             SaveTreeToStream(mem);

             AddBinaryProperty(c45TreeProps, mem.Memory^, mem.Size);
          finally
                 mem.Free;
          end;
     end;
end;

function TC45Classifier.PropTypeOfName(const Name: string): TPropType;
begin
     if (CompareText(Name, c45ClassesProps) = 0) or
        (CompareText(Name, c45TreeProps) = 0)
     then
         Result := ptBinary
     else
         Result := inherited PropTypeOfName(Name);
end;


destructor TC45Classifier.Destroy;
begin
     fTree.Free;

     inherited;
end;

procedure TC45Classifier.LoadTreeFromStream(stream: TStream);
var version : integer;

function ReadFromStream(stream : TStream) : TCustomTreeItem;
var cTag : byte;
begin
     stream.ReadBuffer(cTag, sizeof(byte));
     if cTag = 0 then
     begin
          Result := TTreeNode.Create;
          Result.TreeData := T45NodeData.Create;
          T45NodeData(Result.TreeData).LoadFromStream(stream);
          // left right part
          TTreeNode(Result).LeftItem := ReadFromStream(Stream);
          TTreeNode(Result).RightItem := ReadFromStream(Stream);
     end
     else
     begin
          Result := TTreeLeave.Create;
          Result.TreeData := T45NodeData.Create;
          T45NodeData(Result.TreeData).LoadFromStream(stream);
     end;
end;

begin
     Stream.ReadBuffer(version, sizeof(version));
     assert(version = 0, 'No other version supported');

     fTree := ReadFromStream(Stream);
end;

procedure TC45Classifier.OnLoadBinaryProperty(const Name: String;
  const Value; size: integer);
var mem : TMemoryStream;
begin
     if CompareText(Name, c45ClassesProps) = 0 then
     begin
          if size > 0 then
          begin
               SetLength(fClasses, size div sizeof(LongInt));
               Move(Value, fClasses[0], Length(fClasses));
          end;
     end
     else if CompareText(Name, c45TreeProps) = 0 then
     begin
          Assert(not Assigned(fTree), 'error Tree already assigned');
          mem := TMemoryStream.Create;
          try
             mem.Write(Value, size);
             mem.Position := 0;
             LoadTreeFromStream(mem);
          finally
                 mem.Free;
          end;
     end;
end;

procedure TC45Classifier.SaveTreeToStream(stream: TStream);

procedure WriteNode(node : TCustomTreeItem);
var cTag : byte;
begin
     if node is TTreeNode then
     begin
          cTag := 0;
          stream.Write(ctag, sizeof(byte));
          T45NodeData(node.TreeData).WriteToStream(stream);

          WriteNode(TTreeNode(node).LeftItem);
          WriteNode(TTreeNode(node).RightItem);
     end
     else
     begin
          cTag := 1;
          stream.Write(ctag, sizeof(byte));

          T45NodeData(node.TreeData).WriteToStream(stream);
     end;
end;
var version : integer;
begin
     version := 0;
     Stream.WriteBuffer(version, sizeof(version));
     WriteNode(fTree);
end;

initialization
  RegisterMathIO(TC45Classifier);

end.
