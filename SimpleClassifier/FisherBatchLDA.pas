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

unit FisherBatchLDA;

// #############################################################
// #### Fisher LDA in batch learning mode.
// #############################################################

// based on Martina Urays (and other papers) Phd:
// Incremental, Robust, and Efficient Linear Discriminant Analysis Learning
// http://www.icg.tugraz.at/Members/uray/diplomathesis/diss

interface

uses SysUtils, Classes, BaseClassifier, Matrix, Types, PCA;

// ##################################################
// #### Base batch Fisher LDA learning and classification algorithm
type
  ELDAException = class(ECustomClassifierException);


type
  TFisherAugmentedClassifierType = (ctFast, ctRobust, ctFastRobust);
  
// ##################################################
// #### Base properties
type
  TFisherAugmentedBaseProps = record
    UseFullSpace : boolean;
    ClassifierType : TFisherAugmentedClassifierType;
    NumLDAVectorsToKeep : integer;                   // todo: pherhaps take the pca eigenvalues energy as property
    RobustPCAProps : TFastRobustPCAProps;            // only used in case the classifier type is ctFastRobust
  end;

// ##################################################
// #### Batch Fisher LDA algorithm.
// Phd: Incremental, Robust, and Efficient Linear Discriminant Analysis Learning, page 33
// Augmented subspace learning without any vector reduction
type
  TFisherBatchLDALearner = class(TCustomLearner)
  protected
    fProps : TFisherAugmentedBaseProps;
    fPCA : TMatrixPCA;

    function ExtractMatrixData(var ownsMatrix : boolean) : TDoubleMatrix;
    function PCAAndProject(data : TDoubleMatrix; var U : TDoubleMatrix; var mu : TDoubleMatrix; var A : TDoubleMatrix; var eigVals : TDoubleMatrix) : boolean;
    procedure CreateBaseLDAClassifier(var pcaU, pcaMu, eigVals, ldaV : TDoubleMatrix; var classCenters : TDoubleMatrixDynArr;
                                      const classLabels : TIntegerDynArray; const classIdx : TIntIntArray; numClasses : Integer);
    function DoUnweightedLearn : TCustomClassifier; override;
  public
    procedure SetProperties(const props : TFisherAugmentedBaseProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;

    destructor Destroy; override;
  end;

// ##################################################
// #### Batch Fisher LDA algorithm.
// Phd: Incremental, Robust, and Efficient Linear Discriminant Analysis Learning, page 33
// Augmented subspace learning with vector reduction
type
  TFisherBatchLDAAugmentedBaseLearner = class(TFisherBatchLDALearner)
  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
    procedure ReduceAndAugmentBase(const eigVals : TDoubleMatrix; var ldaV, pcaU : TDoubleMatrix; var theta : double);
  public
    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
  end;

implementation

uses BaseMatrixExamples, Math, LinearAlgebraicEquations, FisherClassifiers, MatrixConst;

{ TFisherBatchLDALearner }

class function TFisherBatchLDALearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := (Classifier = TFisherLDAClassifier) or (Classifier = TFisherRobustLDAClassifier) or (Classifier = TFisherRobustExLDAClassifier);
end;

procedure TFisherBatchLDALearner.SetProperties(
  const props: TFisherAugmentedBaseProps);
begin
     fProps := props;
end;

procedure TFisherBatchLDALearner.CreateBaseLDAClassifier(var pcaU, pcaMu, eigVals,
  ldaV: TDoubleMatrix; var classCenters: TDoubleMatrixDynArr; const classLabels : TIntegerDynArray;
  const classIdx : TIntIntArray; numClasses : Integer);
var matrixData : TDoubleMatrix;
    x, y : integer;
    A : TDoubleMatrix;
    mu : TDoubleMatrix;
    Sb, Sw : TDoubleMatrix;
    tmp, tmpT : TDoubleMatrix;
    sumMtx : TDoubleMatrix;
    regularizeFact : double;
    U, S, V : TDoubleMatrix;
    ownsMatrix : boolean;
begin
     // todo: introduce example weighting
     matrixData := nil;
     mu := nil;
     classCenters := nil;
     sb := nil;
     sw := nil;
     U := nil;
     S := nil;
     V := nil;
     pcaU := nil;
     pcaMu := nil;

     // ###################################################################
     // #### get indexes of the classes and the number of classes
     SetLength(classCenters, numClasses);

     assert(numClasses > 0, 'Error no data');

     // ###################################################################
     // #### shrink the example space such (Fisher LDA)
     try
        matrixData := ExtractMatrixData(ownsMatrix);
        try
           if not PCAAndProject(matrixData, pcaU, pcaMu, A, eigVals) then
              raise ELDAException.Create('Error unable to calculate pca on data');
        finally
               if ownsMatrix then
                  FreeAndNil(matrixData);
        end;

        // ##############################################################
        // #### Calculate Scatter matrices in the feature space
        mu := A.Mean(True);

        for y := 0 to numClasses - 1 do
        begin
             classCenters[y] := TDoubleMatrix.Create(1, mu.Height);

             for x := 0 to Length(classIdx[y]) - 1 do
             begin
                  A.SetSubMatrix(classIdx[y][x], 0, 1, A.Height);
                  classCenters[y].AddInplace(A);
             end;

             classCenters[y].ScaleInPlace(1/Length(classIdx[y]));
        end;

        // between class scatter matrix on A:
        Sb := TDoubleMatrix.Create(classCenters[0].Height, classCenters[0].Height);

        for y := 0 to numClasses - 1 do
        begin
             tmp := classCenters[y].Sub(mu);
             try
                tmpT := tmp.Transpose;
                try
                   tmp.MultInPlace(tmpT);
                   tmp.ScaleInPlace(classIdx[y][0]);
                   sb.AddInplace(tmp);
                finally
                       tmpT.Free;
                end;
             finally
                    tmp.Free;
             end;
        end;

        FreeAndNil(mu);

        // within class scatter matrix on A
        Sw := TDoubleMatrix.Create(classCenters[0].Height, classCenters[0].Height);
        for y := 0 to numClasses - 1 do
        begin
             sumMtx := TDoubleMatrix.Create(classCenters[0].Height, classCenters[0].Height);
             try
                for x := 0 to Length(classIdx[y]) - 1 do
                begin
                     A.SetSubMatrix(classIdx[y][x], 0, 1, A.Height);
                     tmp := A.Sub(classCenters[y]);
                     try
                        //tmp.ScaleInPlace(sqrt(Weights[classIdx[y][x]]));
                        tmp.ScaleInPlace(sqrt(1/Length(classIdx[y])));
                        tmpT := tmp.Transpose;
                        try
                           tmp.MultInPlace(tmpT);
                        finally
                               tmpT.Free;
                        end;
                        sumMtx.AddInplace(tmp);
                     finally
                            tmp.Free;
                     end;
                end;

                Sw.AddInplace(sumMtx);
             finally
                    sumMtx.Free;
             end;
        end;

        FreeAndNil(A);

        // #############################################################
        // #### Perform LDA on A and on the calculated scatter matrices
        // regularize within class scatter matrix which ensures that the matrix can be inverted
        regularizeFact := Sw[0, 0];
        for y := 1 to Sw.Width - 1 do
            regularizeFact := Max(regularizeFact, Sw[y, y]);
        regularizeFact := Max(MinDouble*10000, regularizeFact*0.00001);
        for y := 0 to Sw.Width - 1 do
            Sw[y, y] := Sw[y, y] + regularizeFact;

        // now we can invert the  matrix without problems and calculate the LDA criterium
        if Sw.InvertInPlace <> leOk then
           raise ELDAException.Create('Error singular Sw');
        Sw.MultInPlace(Sb);
        FreeAndNil(Sb);
        if Sw.SVD(U, V, S) <> srOk then
           raise ELDAException.Create('Error LDA matrix cannot be computed');

        FreeAndNil(V);
        FreeAndNil(S);
        FreeAndNil(Sw);

        // the (numClasses - 1) eigenvectors of U matrix contains our LDA vectors
        U.SetSubMatrix(0, 0, numClasses - 1, U.Height);
        ldaV := TDoubleMatrix.Create;
        ldaV.Assign(U, True);
        FreeAndNil(U);

        // project pca class centers to lda space to obtain vi's
        ldaV.TransposeInPlace;
        for y := 0 to numClasses - 1 do
        begin
             tmp := ldaV.Mult(classCenters[y]);
             classCenters[y].Free;
             classCenters[y] := tmp;
        end;
     except
           FreeAndNil(A);

           for y := 0 to numClasses - 1 do
               FreeAndNil(classCenters[y]);

           FreeAndNil(mu);
           FreeAndNil(Sb);
           FreeAndNil(Sw);
           FreeAndNil(U);
           FreeAndNil(S);
           FreeAndNil(V);
           FreeAndNil(pcaU);
           FreeAndNil(pcaMu);
     end;
end;

destructor TFisherBatchLDALearner.Destroy;
begin
     fPCA.Free;

     inherited;
end;

function TFisherBatchLDALearner.DoUnweightedLearn : TCustomClassifier;
var pcaU, pcaMu, ldaV, eigVals: TDoubleMatrix;
    classCenters: TDoubleMatrixDynArr;
    numClasses : integer;
    counter : integer;
    classLabels : TIntegerDynArray;
    classIdx : TIntIntArray;
    distSigmas : TDoubleDynArray;
    clIdx: Integer;
    distances : TDoubleMatrix;
    numCorrect : integer;
    conf : double;
    idx : integer;
begin
     Result := nil;
     eigVals := nil;
     pcaU := nil;
     pcaMu := nil;

     numClasses := IndexOfClasses(classIdx, classLabels);
     // this base classifier does not suppert reduction by now.
     // todo: eventually just don't use the eigenvectors (as proposed in the phd - the LDAonK classifier)
     CreateBaseLDAClassifier(pcaU, pcaMu, eigVals, ldaV, classCenters, classLabels, classIdx, numClasses);

     if fProps.ClassifierType = ctFastRobust then
     begin
          FreeAndNil(pcaU);
          FreeAndNil(pcaMu);
     end;

     eigVals.Free;
     if not Assigned(pcaMu) and (fProps.ClassifierType <> ctFastRobust) then
        raise ELDAException.Create('Error no pca matrix assigned');

     case fProps.ClassifierType of
       ctFast: Result := TFisherLDAClassifier.Create(pcaU, pcaMu, ldaV, classCenters, classLabels);
       ctRobust: Result := TFisherRobustLDAClassifier.Create(pcaU, pcaMu, ldaV, classCenters, classLabels, 0);
       ctFastRobust: Result := TFisherRobustExLDAClassifier.Create(fPCA as TFastRobustPCA, ldaV, classCenters, classLabels);
     end;

     if fProps.ClassifierType = ctFastRobust then
     begin
          pcaU.Free;
          pcaMu.Free;
     end;

     fPCA := nil;

     // #####################################################
     // #### Calculate the distances from the center for all examples
     // -> these are used to calculate the sigmas
     SetLength(distSigmas, numClasses);

     distances := TDoubleMatrix.Create;
     for clIdx := 0 to numClasses - 1 do
     begin
          numCorrect := 0;
          distances.SetWidthHeight(Length(classIdx[clIdx]), 1);
          for counter := 0 to Length(classIdx[clIdx]) - 1 do
          begin
               idx := classIdx[clIdx][counter];
               if Result.Classify( DataSet.Example[ idx ], conf ) = DataSet.Example[ idx ].ClassVal then
               begin
                    distances.vec[numCorrect] := conf;
                    inc(numCorrect);
               end;
          end;

          if numCorrect > 1 then
          begin
               distances.SetSubMatrix(0, 0, numCorrect, 1);
               distances.VarianceInPlace(True);
               distSigmas[clIdx] := distances.Vec[0];
          end;
     end;

     TFisherLDAClassifier(Result).SetSigmaDist(distSigmas);

     distances.Free;
end;

function TFisherBatchLDALearner.ExtractMatrixData(
  var ownsMatrix: boolean): TDoubleMatrix;
var x, y : integer;
begin
     if ( DataSet is TMatrixLearnerExampleList ) and ( TMatrixLearnerExampleList(DataSet).Matrix.Width = DataSet.Count) then
     begin
          Result := TMatrixLearnerExampleList(DataSet).Matrix;
          ownsMatrix := False;
     end
     else
     begin
          // Create a matrix of the feature vectors -> make it compatible with the pca routines
          Result := TDoubleMatrix.Create(DataSet.Count, DataSet[0].FeatureVec.FeatureVecLen);
          for y := 0 to DataSet[0].FeatureVec.FeatureVecLen - 1 do
              for x := 0 to DataSet.Count - 1 do
                  Result[x, y] := DataSet[x].FeatureVec[y];

          ownsMatrix := True;
     end;
end;

function TFisherBatchLDALearner.PCAAndProject(data: TDoubleMatrix; var U, mu,
  A: TDoubleMatrix; var eigVals : TDoubleMatrix): boolean;
var minVal : integer;
    i : integer;
begin
     Result := True;

     minVal := Min(DataSet.Count, DataSet[0].FeatureVec.FeatureVecLen);
     A := TDoubleMatrix.Create(DataSet.Count, minVal);

     if fProps.ClassifierType = ctFastRobust then
     begin
          fpca := TFastRobustPCA.Create([pcaTransposedEigVec]);
          TFastRobustPCA(fpca).SetProperties(fProps.RobustPCAProps);
     end
     else
         fpca := TMatrixPCA.Create([pcaTransposedEigVec]);
     try
        // todo: do we have to use the weights here?
        if not fpca.PCA(data, 1, True) then
        begin
             A.Free;
             Result := False;
             exit;
        end;

        //  project the examples to the PCA space
        // todo: check if the projection needs some kind of weight attention
        for i := 0 to DataSet.Count - 1 do
        begin
             data.SetSubMatrix(i, 0, 1, Data.Height);

             if fProps.ClassifierType = ctFastRobust
             then
                 TFastRobustPCA(fpca).ProjectToFeatureSpaceNonRobust(data, A, i)
             else
                 fpca.ProjectToFeatureSpace(data, A, i);
        end;

        data.UseFullMatrix;

        // ##################################################
        // #### Assign other matrices
        U := TDoubleMatrix.Create;
        U.Assign(fpca.EigVecs);
        mu := TDoubleMatrix.Create;
        mu.Assign(fpca.Mean);

        eigvals := TDoubleMatrix.Create;
        eigVals.Assign(fpca.EigVals);
     finally
            if fProps.ClassifierType <> ctFastRobust then
               FreeAndNil(fpca);
     end;
end;

{ TFisherBatchLDAAugmentedBaseLearner }

class function TFisherBatchLDAAugmentedBaseLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := (Classifier = TFisherLDAClassifier) or (Classifier = TFisherRobustLDAClassifier) or (Classifier = TFisherRobustExLDAClassifier);
end;

function TFisherBatchLDAAugmentedBaseLearner.DoLearn(
  const weights: array of double): TCustomClassifier;
var pcaU, pcaMu, ldaV: TDoubleMatrix;
    classCenters: TDoubleMatrixDynArr;
    numClasses : integer;
    classLabels : TIntegerDynArray;
    classIdx : TIntIntArray;
    eigVals : TDoubleMatrix;
    theta : double;
begin
     Result := nil;
     eigVals := nil;

     numClasses := IndexOfClasses(classIdx, classLabels);
     CreateBaseLDAClassifier(pcaU, pcaMu, eigVals, ldaV, classCenters, classLabels, classIdx, numClasses);

     if not Assigned(pcaMu) then
        raise ELDAException.Create('Error no pca matrix assigned');

     try
        // #########################################################
        // #### Now calculate the reduced and augmented basis

        // split V (1..n) into (v(1..k, v(k - n))
        theta := 0;
        if not fProps.UseFullSpace then
           ReduceAndAugmentBase(eigVals, ldaV, pcaU, theta);

        if fProps.ClassifierType = ctFastRobust then
        begin
             // for the fast robust pca method assign the augmented subspace back to the object
             fPCA.EigVecs.Assign(pcaU);
        end;
     finally
            eigVals.Free;
     end;

     // #######################################################
     // #### Create Classifier
     case fProps.ClassifierType of
       ctFast: Result := TFisherLDAClassifier.Create(pcaU, pcaMu, ldaV, classCenters, classLabels);
       ctRobust: Result := TFisherRobustLDAClassifier.Create(pcaU, pcaMu, ldaV, classCenters, classLabels, theta);
       ctFastRobust: Result := TFisherRobustExLDAClassifier.Create(fPCA as TFastRobustPCA, ldaV, classCenters, classLabels);
     end;

     if fProps.ClassifierType = ctFastRobust then
     begin
          pcaMu.Free;
          pcaU.Free;
     end;

     fPCA := nil;
end;

procedure TFisherBatchLDAAugmentedBaseLearner.ReduceAndAugmentBase(const eigVals : TDoubleMatrix; var ldaV, pcaU: TDoubleMatrix; var theta : double);
var vhat : TDoubleMatrix;
    U, V, W : TDoubleMatrix;
    M : TDoubleMatrix;
    x : integer;
    y : integer;
begin
     ldaV.TransposeInPlace;

     vhat := TDoubleMatrix.Create;
     try
        ldaV.SetSubMatrix(0, fProps.NumLDAVectorsToKeep, ldaV.Width, ldaV.Height - fProps.NumLDAVectorsToKeep);
        vhat.Assign(ldaV, True);
        ldaV.UseFullMatrix;

        // orthogonalize splitted base -> vhatj. The base formula is vhat = v*(v'*v)^-0.5
        if vhat.SVD(U, V, W) <> srOk then
           raise ELDAException.Create('Error orthogonalizing matrix');

        try
           W.Free;

           // v is returned as V transpose
           //V.TransposeInPlace;
           FreeAndNil(vhat);

           // orthogonalize
           vhat := U.Mult(V);
        finally
               U.Free;
               V.Free;
        end;

        // #######################################################
        // #### Transformation matrix: M [ Ik       0k  ]
        //                               [ 0(n-k)   Vhat]
        M := TDoubleMatrix.Create(fProps.NumLDAVectorsToKeep + vhat.Width, fProps.NumLDAVectorsToKeep + vhat.Height);

        for x := 0 to fProps.NumLDAVectorsToKeep - 1 do
            M[x, x] := 1;

        for y := fProps.NumLDAVectorsToKeep to M.Height - 1 do
            for x := fProps.NumLDAVectorsToKeep to M.Width - 1 do
                M[x, y] := vhat[x - fProps.NumLDAVectorsToKeep, y - fProps.NumLDAVectorsToKeep];

        FreeAndNil(vhat);

        // #######################################################
        // #### Augmanted basis for pcaU is already transposed
        pcaU.MultInPlace(M);

        // #######################################################
        // #### Augmanted basis for pcaU:  Uhat = [Uk Un-k*Vhat]
        M.TransposeInPlace;
        M.MultInPlace(ldaV);
        FreeAndNil(ldaV);
        ldaV := M;

        // #######################################################
        // #### estimate expected error distribution.
        // details in: Bischof, Leonardis: Robust Recognition using Eigenimages
        theta := 0;
        for y := fProps.NumLDAVectorsToKeep to eigVals.Height - 1 do
            theta := theta + eigVals[0, y];
        theta := theta*2/(eigVals.Height);

        // backtranspose
        ldaV.TransposeInPlace;
     except
           FreeAndNil(vhat);
           FreeAndNil(M);

           raise;
     end;
end;

end.
