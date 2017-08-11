//
// Created by Vincent_Bode on 29.06.2017.
//

#include <ctime>
#include <mpi.h>
#include <cstring>

#include <sgpp/datadriven/application/work_in_progress/LearnerSGDEOnOffParallel.hpp>
#include <sgpp/datadriven/application/work_in_progress/NetworkMessageData.hpp>
#include <sgpp/datadriven/application/work_in_progress/MPIMethods.hpp>
#include <sgpp/base/exception/application_exception.hpp>
#include <thread>
#include <numeric>


namespace sgpp {
    namespace datadriven {

        //Pending MPI Requests
        std::list<PendingMPIRequest> MPIMethods::pendingMPIRequests;
        MPIRequestPool MPIMethods::mpiRequestStorage;
        int MPIMethods::mpiWorldSize = -1;
        LearnerSGDEOnOffParallel *MPIMethods::learnerInstance;


        bool MPIMethods::isMaster() {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank == MPI_MASTER_RANK;
        }

        void MPIMethods::initMPI(LearnerSGDEOnOffParallel *learnerInstance) {
            MPI_Init(nullptr, nullptr);

            // Get World Size
            MPI_Comm_size(MPI_COMM_WORLD, &mpiWorldSize);

            //Get Rank
            int world_rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

            //Get Processor Name
            char mpiProcessorName[MPI_MAX_PROCESSOR_NAME_LENGTH];
            int nameLength;
            MPI_Get_processor_name(mpiProcessorName, &nameLength);

            std::cout << "Processor " << mpiProcessorName << " (rank " << world_rank
                      << ") has joined MPI pool of size " << mpiWorldSize << std::endl;

            //Setup receiving messages from master/workers
            {
                auto *mpiPacket = new MPI_Packet;
                auto &unicastInputRequest = createPendingMPIRequest(mpiPacket);
                unicastInputRequest.disposeAfterCallback = false;
                unicastInputRequest.callback = [](PendingMPIRequest &request) {
                    std::cout << "Incoming MPI unicast" << std::endl;
                    processIncomingMPICommands(request.buffer);

                    std::cout << "Zeroing MPI Request" << std::endl;
                    std::memset(request.getMPIRequestHandle(), 0, sizeof(MPI_Request));

                    std::cout << "Zeroing Buffer" << std::endl;
                    std::memset(request.buffer, 0, sizeof(MPI_Packet));


                    std::cout << "Restarting irecv request." << std::endl;
                    MPI_Irecv(request.buffer, sizeof(MPI_Packet), MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE,
                              MPI_ANY_TAG, MPI_COMM_WORLD, request.getMPIRequestHandle());
                };

                MPI_Irecv(mpiPacket, sizeof(MPI_Packet), MPI_UNSIGNED_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG,
                          MPI_COMM_WORLD,
                          unicastInputRequest.getMPIRequestHandle());

                std::cout << "Started listening for unicasts from any sources" << std::endl;
            }
            if (!isMaster()) {
                auto *mpiPacket = new MPI_Packet;
                auto &broadcastInputRequest = createPendingMPIRequest(mpiPacket);
                broadcastInputRequest.disposeAfterCallback = false;
                broadcastInputRequest.callback = [](PendingMPIRequest &request) {
                    std::cout << "Incoming MPI broadcast" << std::endl;
                    processIncomingMPICommands(request.buffer);

                    std::cout << "Zeroing MPI Request" << std::endl;
                    std::memset(request.getMPIRequestHandle(), 0, sizeof(MPI_Request));

                    std::cout << "Zeroing Buffer" << std::endl;
                    std::memset(request.buffer, 0, sizeof(MPI_Packet));

                    std::cout << "Restarting ibcast request." << std::endl;
                    MPI_Ibcast(request.buffer, sizeof(MPI_Packet), MPI_UNSIGNED_CHAR, MPI_MASTER_RANK,
                               MPI_COMM_WORLD, request.getMPIRequestHandle());
                };
                MPI_Ibcast(mpiPacket, sizeof(MPI_Packet), MPI_UNSIGNED_CHAR, MPI_MASTER_RANK, MPI_COMM_WORLD,
                           broadcastInputRequest.getMPIRequestHandle());

                std::cout << "Started listening for broadcasts from task master" << std::endl;
            }

            MPIMethods::learnerInstance = learnerInstance;
        }

        void MPIMethods::synchronizeBarrier() {
            MPI_Barrier(MPI_COMM_WORLD);
        }

        void MPIMethods::sendGridComponentsUpdate(std::vector<RefinementResult> *refinementResults) {

            std::cout << "Updating the grid components on workers..." << std::endl;

            for (size_t classIndex = 0; classIndex < refinementResults->size(); classIndex++) {
                RefinementResult refinementResult = (*refinementResults)[classIndex];

                std::cout << "Updating grid for class " << classIndex
                          << " (" << refinementResult.addedGridPoints.size() << " additions, "
                          << refinementResult.deletedGridPointsIndexes.size() << " deletions)" << std::endl;

                sendRefinementUpdates(classIndex, refinementResult.deletedGridPointsIndexes,
                                      refinementResult.addedGridPoints);
            }

            std::cout << "Finished updating the grid components on workers." << std::endl;
        }

        void MPIMethods::sendRefinementUpdates(size_t &classIndex, std::list<size_t> &deletedGridPointsIndexes,
                                               std::list<LevelIndexVector> &addedGridPoints) {
            // Deleted grid points
            {
                std::list<size_t>::const_iterator iterator = deletedGridPointsIndexes.begin();
                std::list<size_t>::const_iterator listEnd = deletedGridPointsIndexes.end();
                while (iterator != listEnd) {
                    auto *mpiPacket = new MPI_Packet;
                    mpiPacket->commandID = UPDATE_GRID;

                    auto *networkMessage = (RefinementResultNetworkMessage *) mpiPacket->payload;

                    networkMessage->classIndex = classIndex;
                    networkMessage->updateType = DELETED_GRID_POINTS_LIST;
                    networkMessage->gridversion = learnerInstance->getCurrentGridVersion();

                    size_t numPointsInBuffer = fillBufferWithData<std::list<size_t>::const_iterator>(
                            (void *) networkMessage->payload,
                            (void *) std::end(networkMessage->payload),
                            iterator,
                            listEnd);
                    networkMessage->listLength = numPointsInBuffer;

                    std::cout << "Sending updated for class " << networkMessage->classIndex
                              << " with " << networkMessage->listLength
                              << " deletions" << std::endl;

                    sendIBcast(mpiPacket);
                }
            }
            // Added grid points
            {
                auto iterator = addedGridPoints.begin();
                std::list<LevelIndexVector>::const_iterator listEnd = addedGridPoints.end();
                while (iterator != listEnd) {
                    auto *mpiPacket = new MPI_Packet;
                    mpiPacket->commandID = UPDATE_GRID;

                    auto *networkMessage = (RefinementResultNetworkMessage *) mpiPacket->payload;

                    networkMessage->classIndex = classIndex;
                    networkMessage->updateType = ADDED_GRID_POINTS_LIST;
                    networkMessage->gridversion = learnerInstance->getCurrentGridVersion();

                    size_t numPointsInBuffer = fillBufferWithLevelIndexData(networkMessage->payload,
                                                                            std::end(networkMessage->payload), iterator,
                                                                            listEnd);
                    networkMessage->listLength = numPointsInBuffer;

                    std::cout << "Sending updated for class " << networkMessage->classIndex
                              << " with " << networkMessage->listLength
                              << " additions" << std::endl;

                    sendIBcast(mpiPacket);
                }
            }
        }

        void MPIMethods::bcastCommandNoArgs(MPI_COMMAND_ID commandId) {
            auto *mpiPacket = new MPI_Packet;
            mpiPacket->commandID = commandId;

            sendIBcast(mpiPacket);
        }

        void MPIMethods::sendCommandNoArgs(const int destinationRank, MPI_COMMAND_ID commandId) {
            auto *mpiPacket = new MPI_Packet;
            mpiPacket->commandID = commandId;

            sendISend(destinationRank, mpiPacket);
        }

        void MPIMethods::sendIBcast(MPI_Packet *mpiPacket) {
            PendingMPIRequest &pendingMPIRequest = createPendingMPIRequest(mpiPacket);

            MPI_Ibcast(mpiPacket, sizeof(MPI_Packet), MPI_UNSIGNED_CHAR, MPI_MASTER_RANK,
                       MPI_COMM_WORLD, pendingMPIRequest.getMPIRequestHandle());

            std::cout << "Ibcast request stored at " << &pendingMPIRequest << std::endl;
        }

        void MPIMethods::sendISend(const int destinationRank, MPI_Packet *mpiPacket) {
            PendingMPIRequest &pendingMPIRequest = createPendingMPIRequest(mpiPacket);

            //Point to the request in vector instead of stack
            MPI_Isend(mpiPacket, sizeof(MPI_Packet), MPI_UNSIGNED_CHAR, destinationRank, COMMAND_TAG,
                      MPI_COMM_WORLD, pendingMPIRequest.getMPIRequestHandle());

            std::cout << "Isend request stored at " << &pendingMPIRequest << std::endl;
        }

        PendingMPIRequest &MPIMethods::createPendingMPIRequest(MPI_Packet *mpiPacket) {
            pendingMPIRequests.emplace_back(&mpiRequestStorage);
            PendingMPIRequest &pendingMPIRequest = pendingMPIRequests.back();
            pendingMPIRequest.disposeAfterCallback = true;
            pendingMPIRequest.callback = [](PendingMPIRequest &request) {
                std::cout << "Pending MPI request " << &request << " completed." << std::endl;
            };
            pendingMPIRequest.buffer = mpiPacket;
            return pendingMPIRequest;
        }

        //TODO: This was imported from Merge
        size_t MPIMethods::receiveMergeGridNetworkMessage(MergeGridNetworkMessage &networkMessage) {
            if (networkMessage.gridversion != learnerInstance->getCurrentGridVersion()) {
                std::cout << "Received merge grid request with incorrect grid version!"
                          << " local: " << learnerInstance->getCurrentGridVersion()
                          << ", remote: " << networkMessage.gridversion
                          << std::endl;
                std::cout << "!#!#!# IGNORING ERROR #!#!#!" << std::endl;
                return 0;
            }

            base::DataVector alphaVector(networkMessage.payloadLength);

            auto *payload = (double *) networkMessage.payload;
            for (size_t index = 0; index < networkMessage.payloadLength; index++) {
                alphaVector[networkMessage.payloadOffset + index] = payload[index];
            }

            learnerInstance->mergeAlphaValues(networkMessage.classIndex, alphaVector, networkMessage.batchSize);

            std::cout << "Updated alpha values from network message offset " << networkMessage.payloadOffset
                      << ", class " << networkMessage.classIndex
                      << ", length " << networkMessage.payloadLength << std::endl;
        }

        //TODO: This was imported from Merge
/*
        size_t MPIMethods::sendRefinementResultPacket(size_t classIndex, RefinementResultsUpdateType updateType,
                                                      const RefinementResult &refinementResult, int offset,
                                                      std::list::iterator &iterator) {
            MPI_Packet *mpiPacket = new MPI_Packet;
            RefinementResultNetworkMessage *networkMessage = (RefinementResultNetworkMessage *) mpiPacket->payload;

            networkMessage->classIndex = classIndex;
            networkMessage->gridversion = refinementResult.gridversion;
            networkMessage->updateType = updateType;

            //Minimum between remaining grid points and maximum number of grid points that will still fit
            size_t numPointsInPacket = 0;

            if (updateType == DELETED_GRID_POINTS_LIST) {
                size_t endOfPayload = sizeof(RefinementResultNetworkMessage::payload) / sizeof(int);

                //Copy data from list into network buffer
                int *deletedGridPoints = (int *) networkMessage->payload;
                for (int bufferIndex = 0; bufferIndex < endOfPayload; bufferIndex++) {
                    deletedGridPoints[bufferIndex] = *iterator;
                    iterator++;
                    numPointsInPacket++;
                }
            } else if (updateType == ADDED_GRID_POINTS_LIST) {
                //TODO: These vectors do not have a constant grid size
                size_t endOfPayload = sizeof(RefinementResultNetworkMessage::payload) / sizeof(sgpp::base::DataVector);
                //Copy data from list into network buffer
                sgpp::base::DataVector *addedGridPoints = (sgpp::base::DataVector *) networkMessage->payload;
                for (int bufferIndex = 0; bufferIndex < endOfPayload; bufferIndex++) {
                    addedGridPoints[bufferIndex] = *iterator;
                    iterator++;
                    numPointsInPacket++;
                }
            }

            networkMessage->listLength = numPointsInPacket;

            printf("Sending update for class %zu with %i modifications", classIndex,
                   networkMessage->listLength);
            PendingMPIRequest pendingMPIRequest;
            pendingMPIRequest.buffer = mpiPacket;
            //TODO: DANGEROUS
            pendingMPIRequests.push_back(pendingMPIRequest);

            //Send the smallest packet possible
            MPI_Ibcast(&networkMessage, numPointsInPacket + 3, MPI_UNSIGNED_CHAR, MPI_MASTER_RANK,
                       MPI_COMM_WORLD, &(pendingMPIRequest.request));
            return numPointsInPacket;
        }
*/

        size_t
        MPIMethods::sendMergeGridNetworkMessage(size_t classIndex, size_t batchSize, base::DataVector &alphaVector) {
            size_t offset = 0;
            while (offset < alphaVector.size()) {
                auto *mpiPacket = new MPI_Packet;
                mpiPacket->commandID = MERGE_GRID;

                void *payloadPointer = &(mpiPacket->payload);

                auto *networkMessage = static_cast<MergeGridNetworkMessage *>(payloadPointer);

                networkMessage->classIndex = classIndex;
                networkMessage->gridversion = learnerInstance->getCurrentGridVersion();
                networkMessage->payloadOffset = offset;
                networkMessage->batchSize = batchSize;

                size_t numPointsInPacket = 0;

                auto beginIterator = alphaVector.begin();
                auto endIterator = alphaVector.end();

                numPointsInPacket = fillBufferWithData(networkMessage->payload, std::end(networkMessage->payload),
                                                       beginIterator, endIterator);

                networkMessage->payloadLength = numPointsInPacket;

                std::cout << "Sending merge for class " << classIndex
                          << " offset " << offset
                          << " with " << numPointsInPacket << " values"
                          << " and grid version " << networkMessage->gridversion << std::endl;
                std::cout << "Alpha sum is " << std::accumulate(alphaVector.begin(), alphaVector.end(), 0.0)
                          << std::endl;
                sendISend(MPI_MASTER_RANK, mpiPacket);
                offset += numPointsInPacket;
            }

            return offset;
        }


        //TODO: !!!!!!!!! Compiler Errors

        //TODO: Ensure compiler calls the correct method
        size_t
        MPIMethods::fillBufferWithLevelIndexData(void *buffer, const void *bufferEnd,
                                                 std::list<LevelIndexVector>::iterator &iterator,
                                                 std::list<LevelIndexVector>::const_iterator &listEnd) {
            size_t copiedValues = 0;
            auto *bufferPointer = static_cast<unsigned long *>(buffer);
            while (iterator != listEnd && bufferPointer < bufferEnd) {
                LevelIndexVector levelIndexVector = *iterator;
                if (bufferPointer + 2 * levelIndexVector.size() * sizeof(unsigned long) >= bufferEnd) {
                    break;
                }
                for (auto &levelIndexPair : levelIndexVector) {
                    *bufferPointer = levelIndexPair.level;
                    bufferPointer++;
                    *bufferPointer = levelIndexPair.index;
                    bufferPointer++;
                }
                copiedValues++;
                iterator++;
            }
            return copiedValues;
        }

        template<typename Iterator>
        size_t
        MPIMethods::fillBufferWithData(void *buffer, void *bufferEnd, Iterator &iterator,
                                       Iterator &listEnd) {
            auto *bufferPointer = (typename std::iterator_traits<Iterator>::value_type *) buffer;
            size_t copiedValues = 0;
            while (bufferPointer + sizeof(typename std::iterator_traits<Iterator>::value_type) < bufferEnd &&
                   iterator != listEnd) {
                *bufferPointer = *iterator;

                bufferPointer++;
                iterator++;
                copiedValues++;
            }
            return copiedValues;
        }

        void MPIMethods::processCompletedMPIRequests() {

            std::cout << "Checking " << pendingMPIRequests.size() << " pending MPI requests" << std::endl;
            auto pendingMPIRequestIterator = pendingMPIRequests.end();
            auto listBegin = pendingMPIRequests.begin();

            // In order to process the send requests first we start from the back

            while (pendingMPIRequestIterator != listBegin) {
                pendingMPIRequestIterator--;

                MPI_Status mpiStatus{};
                int operationCompleted;

                std::cout << "Testing request " << &*pendingMPIRequestIterator << std::endl;
                if (MPI_Test(pendingMPIRequestIterator->getMPIRequestHandle(), &operationCompleted, &mpiStatus) != 0) {
                    std::cout << "Error MPI Test reported" << std::endl;
                    exit(-1);
                }
                std::cout << "Pending request has status " << operationCompleted
                          << " MPI_ERROR: " << mpiStatus.MPI_ERROR
                          << " MPI SOURCE: " << mpiStatus.MPI_SOURCE
                          << " MPI TAG: " << mpiStatus.MPI_TAG << std::endl;
                if (operationCompleted != 0) {

                    processCompletedMPIRequest(pendingMPIRequestIterator);


                    std::cout << "Relaunching processCompletedMPIRequests" << std::endl;
                    processCompletedMPIRequests();
                    break;
                }
            }
        }

        void MPIMethods::processCompletedMPIRequest(
                const std::list<sgpp::datadriven::PendingMPIRequest>::iterator &pendingMPIRequestIterator) {
            std::cout << "Executing callback" << std::endl;
            //Execute the callback
            pendingMPIRequestIterator->callback(*pendingMPIRequestIterator);
            std::cout << "Callback complete" << std::endl;


            if (pendingMPIRequestIterator->disposeAfterCallback) {
                        //TODO Deleting a void pointer here
                        std::cout << "Attempting to delete pending mpi request" << std::endl;
                        delete[] pendingMPIRequestIterator->buffer;

                        pendingMPIRequests.erase(pendingMPIRequestIterator);
                        std::cout << "Deleted pending mpi request" << std::endl;

                    } else {
//                        std::cout << "Zeroing MPI Request" << std::endl;
//                        std::memset(pendingMPIRequestIterator->request, 0, sizeof(MPI_Request));
//
//                        //TODO: This is done after the request has been re-launched
//                        std::cout << "Zeroing Buffer" << std::endl;
//                        std::memset(pendingMPIRequestIterator->buffer, 0, sizeof(MPI_Packet));

                    }
        }

        void MPIMethods::waitForAllMPIRequestsToComplete() {
            for (PendingMPIRequest &pendingMPIRequest : pendingMPIRequests) {
                MPI_Wait(pendingMPIRequest.getMPIRequestHandle(), MPI_STATUS_IGNORE);
            }
        }

        void MPIMethods::waitForAnyMPIRequestsToComplete() {
            int completedRequest;
            MPI_Status mpiStatus{};
            std::cout << "Waiting for " << pendingMPIRequests.size() << " MPI requests to complete" << std::endl;
            MPI_Waitany(pendingMPIRequests.size(), mpiRequestStorage.getMPIRequests(), &completedRequest,
                        &mpiStatus);
            std::cout << "MPI request " << completedRequest << " completed"
                      << " MPI_ERROR: " << mpiStatus.MPI_ERROR
                      << " MPI SOURCE: " << mpiStatus.MPI_SOURCE
                      << " MPI TAG: " << mpiStatus.MPI_TAG << std::endl;
            processCompletedMPIRequest(std::next(pendingMPIRequests.begin(), completedRequest));
        }

        void MPIMethods::receiveGridComponentsUpdate(RefinementResultNetworkMessage *networkMessage) {
            unsigned long classIndex = networkMessage->classIndex;
            RefinementResult &refinementResult = learnerInstance->getRefinementResult(classIndex);

            size_t listLength = networkMessage->listLength;
            size_t processedPoints = 0;
            void *bufferEnd = std::end(networkMessage->payload);

            std::cout << "Receiving " << listLength << " grid modifications for class " << classIndex
                      << std::endl;

            switch (networkMessage->updateType) {
                case DELETED_GRID_POINTS_LIST: {
                    auto *bufferIterator = (size_t *) networkMessage->payload;
                    while (bufferIterator < bufferEnd && processedPoints < listLength) {
                        refinementResult.deletedGridPointsIndexes.push_back(*bufferIterator);
                        bufferIterator++;
                        processedPoints++;
                    }
                }
                    break;
                case ADDED_GRID_POINTS_LIST: {

                    auto *bufferIterator = (unsigned long *) networkMessage->payload;
                    size_t dimensionality = learnerInstance->getDimensionality();
                    while (bufferIterator < bufferEnd && processedPoints < listLength) {
                        LevelIndexVector dataVector(dimensionality);
                        size_t currentDimension = 0;
                        while (currentDimension < dimensionality && bufferIterator < bufferEnd) {
                            dataVector[currentDimension].level = *bufferIterator;
                            bufferIterator++;
                            dataVector[currentDimension].index = *bufferIterator;
                            bufferIterator++;

                            currentDimension++;
                        }
                        refinementResult.addedGridPoints.push_back(dataVector);
                        processedPoints++;
                    }
                    break;
                }
            }
            std::cout << "Updated refinement result " << classIndex << " ("
                      << refinementResult.addedGridPoints.size()
                      << " additions, "
                      << refinementResult.deletedGridPointsIndexes.size() <<
                      " deletions)" << std::endl;
            learnerInstance->updateClassVariablesAfterRefinement(&refinementResult,
                                                                 learnerInstance->getDensityFunctions()[classIndex].first.get());
            learnerInstance->setLocalGridVersion(networkMessage->gridversion);
        }

        void MPIMethods::processIncomingMPICommands(MPI_Packet *mpiPacket) {
            std::cout << "Processing incoming command " << mpiPacket->commandID << std::endl;
            void *networkMessagePointer = &(mpiPacket->payload);

            switch (mpiPacket->commandID) {
                case START_SYNCHRONIZE_PACKETS:
                    startSynchronizingPackets();
                    break;
                case END_SYNCHRONIZE_PACKETS:
                    endSynchronizingPackets();
                    break;
                case UPDATE_GRID: {
                    auto *refinementResultNetworkMessage = static_cast<RefinementResultNetworkMessage *>(networkMessagePointer);
                    receiveGridComponentsUpdate(refinementResultNetworkMessage);
                }
                    break;
                case MERGE_GRID: {
                    auto *mergeGridNetworkMessage = static_cast<MergeGridNetworkMessage *>(networkMessagePointer);
                    receiveMergeGridNetworkMessage(*mergeGridNetworkMessage);
                }
                    break;
                case ASSIGN_BATCH:
                    runBatch(mpiPacket);
                    break;
                case SHUTDOWN:
                    std::cout << "Worker shutdown requested" << std::endl;
                    learnerInstance->shutdown();
                    break;
                case NULL_COMMAND:
                    std::cout << "Error: Incoming command has undefined command id" << std::endl;
                    exit(-1);
                default:
                    std::cout << "Error: MPI unknown command id: " << mpiPacket->commandID << std::endl;
                    exit(-1);
            }
        }

        void MPIMethods::startSynchronizingPackets() {
            //TODO
            std::cout << "Synchronizing not implemented" << std::endl;
        }

        void MPIMethods::finalizeMPI() {
            MPI_Finalize();
        }

        void MPIMethods::endSynchronizingPackets() {
            //TODO
            std::cout << "Synchronizing not implemented" << std::endl;
        }

        void MPIMethods::assignBatch(const int workerID, size_t batchOffset, size_t batchSize, bool doCrossValidation) {
            auto *mpiPacket = new MPI_Packet;
            mpiPacket->commandID = ASSIGN_BATCH;

            auto *message = (AssignBatchNetworkMessage *) mpiPacket->payload;
            message->batchOffset = batchOffset;
            message->batchSize = batchSize;
            message->doCrossValidation = doCrossValidation;

            sendISend(workerID, mpiPacket);
        }

        void MPIMethods::runBatch(MPI_Packet *mpiPacket) {
            auto *message = (AssignBatchNetworkMessage *) mpiPacket->payload;
            std::cout << "runbatch dim " << learnerInstance->getDimensionality() << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "creating dataset" << std::endl;
            Dataset dataset{message->batchSize, learnerInstance->getDimensionality()};
            learnerInstance->workBatch(dataset, message->batchOffset, message->doCrossValidation);
        }

        int MPIMethods::getWorldSize() {
            return mpiWorldSize;
        }

        size_t MPIMethods::getQueueSize() {
            return pendingMPIRequests.size();
        }

    }
}