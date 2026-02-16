export interface ClassifierRuntimeMode {
    enableCandidate: boolean;
    enableShadow: boolean;
    fallbackToOcc: boolean;
    strictCandidate: boolean;
    stageBUnresolvedPreferInside?: boolean;
}

export interface ClassifierRuntimeSnapshot {
    enableCandidate: boolean | undefined;
    enableShadow: boolean | undefined;
    fallbackToOcc: boolean | undefined;
    strictCandidate: boolean | undefined;
    stageBUnresolvedPreferInside: boolean | undefined;
}

type RuntimeGlobals = typeof globalThis & {
    __ENABLE_LOCAL_UV_CLASSIFIER_CANDIDATE__?: boolean;
    __ENABLE_LOCAL_UV_CLASSIFIER_SHADOW__?: boolean;
    __LOCAL_UV_CLASSIFIER_FALLBACK_TO_OCC__?: boolean;
    __LOCAL_UV_CLASSIFIER_CANDIDATE_STRICT__?: boolean;
    __LOCAL_UV_STAGEB_UNRESOLVED_PREFER_INSIDE__?: boolean;
};

function setBooleanGlobal(key: keyof RuntimeGlobals, value: boolean | undefined): void {
    const globals = globalThis as RuntimeGlobals;
    if (value === undefined) {
        delete globals[key];
    } else {
        globals[key] = value;
    }
}

export function snapshotClassifierRuntimeGlobals(): ClassifierRuntimeSnapshot {
    const globals = globalThis as RuntimeGlobals;
    return {
        enableCandidate: globals.__ENABLE_LOCAL_UV_CLASSIFIER_CANDIDATE__,
        enableShadow: globals.__ENABLE_LOCAL_UV_CLASSIFIER_SHADOW__,
        fallbackToOcc: globals.__LOCAL_UV_CLASSIFIER_FALLBACK_TO_OCC__,
        strictCandidate: globals.__LOCAL_UV_CLASSIFIER_CANDIDATE_STRICT__,
        stageBUnresolvedPreferInside: globals.__LOCAL_UV_STAGEB_UNRESOLVED_PREFER_INSIDE__,
    };
}

export function applyClassifierRuntimeGlobals(mode: ClassifierRuntimeMode): void {
    setBooleanGlobal("__ENABLE_LOCAL_UV_CLASSIFIER_CANDIDATE__", mode.enableCandidate);
    setBooleanGlobal("__ENABLE_LOCAL_UV_CLASSIFIER_SHADOW__", mode.enableShadow);
    setBooleanGlobal("__LOCAL_UV_CLASSIFIER_FALLBACK_TO_OCC__", mode.fallbackToOcc);
    setBooleanGlobal("__LOCAL_UV_CLASSIFIER_CANDIDATE_STRICT__", mode.strictCandidate);
    if (mode.stageBUnresolvedPreferInside !== undefined) {
        setBooleanGlobal("__LOCAL_UV_STAGEB_UNRESOLVED_PREFER_INSIDE__", mode.stageBUnresolvedPreferInside);
    }
}

export function restoreClassifierRuntimeGlobals(snapshot: ClassifierRuntimeSnapshot): void {
    setBooleanGlobal("__ENABLE_LOCAL_UV_CLASSIFIER_CANDIDATE__", snapshot.enableCandidate);
    setBooleanGlobal("__ENABLE_LOCAL_UV_CLASSIFIER_SHADOW__", snapshot.enableShadow);
    setBooleanGlobal("__LOCAL_UV_CLASSIFIER_FALLBACK_TO_OCC__", snapshot.fallbackToOcc);
    setBooleanGlobal("__LOCAL_UV_CLASSIFIER_CANDIDATE_STRICT__", snapshot.strictCandidate);
    setBooleanGlobal("__LOCAL_UV_STAGEB_UNRESOLVED_PREFER_INSIDE__", snapshot.stageBUnresolvedPreferInside);
}
