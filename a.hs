{-# LANGUAGE BangPatterns, DataKinds, KindSignatures, FlexibleInstances #-}

import Control.DeepSeq

import Control.Monad

import qualified Data.ByteString.Lazy as BS

import qualified Data.List as List

import qualified Data.Vector.Unboxed as UV
import qualified Data.Vector.Unboxed.Mutable as UVM

import System.IO

import GHC.TypeLits
import qualified Numeric.LinearAlgebra.Data as LAD
import qualified Numeric.LinearAlgebra.Static as LA

import Debug.Trace

coefsSize :: Int
coefsSize = 57 -- 29 or 57.

extendedRowSize :: Int
extendedRowSize = coefsSize + 1 -- label and constant.

-- |Quick and dirty conjugate gradient implementation.
cg :: KnownNat n => LA.L n n -> LA.R n -> [LA.R n]
cg a b
	| otherwise = loop (b - (a LA.#> b)) b
	where
		loop p x
			| r2 < 1e-4 = [x]
			| otherwise =
				x : loop p' x'
			where
				r = b - (a LA.#> x)
				r2 = r LA.<.> r
				ap = a LA.#> p
				pAp = p LA.<.> ap
				alpha = r LA.<.> p / pAp
				x' = x + LA.konst alpha * p
				r' = r - LA.konst alpha * ap
				beta = (r' LA.<.> r') / r2
				p' = r' + LA.konst beta * p

sigmoid :: Double -> Double
sigmoid x = 1/(1 + exp (negate x))

type N = 57

toSample :: UV.Vector Double -> (Double, LA.R N)
toSample vec = force (label, LA.fromList inputs)
	where
		dbls = UV.toList vec
		label : inputs' = dbls
		inputs =
			--inputs' ++ [1]
			inputs'

evalOnInputs :: LA.R N -> LA.R N -> Double
evalOnInputs a b = min (1-eps) $ max eps $ sigmoid $ a LA.<.> b
	where
		eps = 1e-12

evaluateGLN :: LA.R N -> [UV.Vector Double] -> IO Double
evaluateGLN coefs csvRows = do
	let	labelInputs = map toSample csvRows
		outs = map (evalOnInputs coefs . snd) labelInputs
		right (label, _) out = (label > 0.5) == (out > 0.5)
		rights = zipWith right labelInputs outs
		rightsCount = sum $ map fromEnum rights
		lossFunc (label, _) out = negate $ (/ log 2) $ label * log out + (1 - label) * log (1 - out)
		loss = sum (zipWith lossFunc labelInputs outs) / fromIntegral (length labelInputs)
		total = length labelInputs
	putStrLn $ "evaluated: "++show rightsCount++" from "++show total++", ratio "++show (fromIntegral rightsCount / fromIntegral total)++", loss "++show loss
	return loss

trainGLN :: [UV.Vector Double] -> IO (LA.R N)
trainGLN labelInputs = loop 1e100 (LA.konst 1 / LA.dim)
	where
		evalAccumulate coefs (!matrA, !colB) csvRow =
			(matrA + a, colB + b)
			where
				(expected, inputs) = toSample csvRow
				y = evalOnInputs inputs coefs
				w = y * (1 - y)
				wi = LA.konst w * inputs
				a = LA.outer inputs wi
				b = wi * LA.konst (expected - y)
		loop prevLoss coefs = do
			thisLoss <- evaluateGLN coefs' labelInputs
			if thisLoss > prevLoss - 1e-6
				then return coefs'
				else loop thisLoss coefs'
			where
				(a, b) = List.foldl' (evalAccumulate coefs) (LA.konst 0, LA.konst 0) labelInputs
				coefs' = coefs + last (cg a b)

-- |Read dataset with possible extension of features.
readDataSet :: Int -> [String] -> IO [UV.Vector Double]
readDataSet numLines lines = do
	m <- UVM.new $ numLines * extendedRowSize
	readLoop m 0 lines
	putStrLn $ "read whole "++show numLines
	dsv <- UV.freeze m
	return [UV.slice (i * extendedRowSize) extendedRowSize dsv | i <- [0..numLines-1]]
	where
		extend = case coefsSize of
			29 -> False
			57 -> True
			_ -> error "coefsSize must be 29 or 57"
		readLoop m i []
			| i < numLines = error "premature EOF???"
			| otherwise = return ()
		readLoop m i (l:ls)
			| i >= numLines = return ()
			| otherwise = do
				forM_ (zip [0..coefsSize-1] extendedRow) $ \(j, e) -> do
					let	k = i * extendedRowSize + j
					UVM.write m k e
				when (mod i 100000 == 0) $ putStrLn $ "read "++show i
				readLoop m (i+1) ls
			where
				dbls@(label:inputs) = read $ "["++l++"]"
				extendedRow = dbls ++ (if extend then map (^2) inputs else []) ++ [1]

main = do
	let	m = 100000 -- set to 100000 for full dataset.
		trainPart = 105 * m
		testPart = 5 * m
	ls <- lines <$> readFile "HIGGS.csv"
	dataset <- readDataSet (trainPart + testPart) ls
	let	(toTrain, toTest) = splitAt trainPart dataset
	coefs <- trainGLN toTrain
	evaluateGLN coefs toTest
	return ()

t = main

